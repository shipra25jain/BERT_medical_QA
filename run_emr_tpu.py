from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import os
import random
import re

import enum
from bert import modeling
from bert import optimization
from bert import tokenization
import numpy as np
import tensorflow as tf
import string
import sys

"""script to train the BERT based model for machine comprehension task on emrQA dataset (electronic medical records) on TPUs"""

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("train_precomputed_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_integer("train_num_precomputed", None,
                     "Number of precomputed tf records for training.")

flags.DEFINE_string("eval_precomputed_file", None,
                    "Precomputed tf records for evaluation. The records for evaluation are different from the records for prediction in the sense that evaluation records store labels as well to compute the loss")

flags.DEFINE_integer("eval_num_precomputed", None,
                     "Number of precomputed tf records for evaluation.")
flags.DEFINE_string("predict_precomputed_file", None,
                    "Precomputed tf records for prediction. Doesnot contain labels")

flags.DEFINE_integer("predict_num_precomputed", None,
                     "Number of precomputed tf records for prediction.")

flags.DEFINE_string(
    "predict_file", None,
    "Emr json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "output_prediction_file", None,
    "Where to print predictions in EMR prediction format, to be passed to")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model) or could be finetuned squad or trained LM as well.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

""" have used max_seq_length as 512 for emr dataset as clinical notes are pretty long"""
flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 384,
    "When splitting up a long document into small windows, how much stride to "
    "take between consecutive windows.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for evaluation.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")
"""recommended to keep iterations_per_loop equal to save_checkpoints_steps"""

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. ")

TextSpan = collections.namedtuple("TextSpan", "token_positions text")

_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)



def get_examples(input_json):
    """ reads the train/test/validation json to give the examples, where each example is a paragraph with multiple question-answer instances """
    with open(input_json, "r") as f:
        data = json.load(f)
    for j in range(len(data['paragraphs'])):
        yield data['paragraphs'][j]


def read_examples(input_json, is_training):
    """ reads input json to return list of examples where each example has context paragraph, question and answer span(if is_training is true)"""
    examples = []
    for entry in get_examples(input_json):
        examples.extend(read_emr_entry(entry, is_training))
    return examples


def read_emr_examples(input_json, is_training):
    """ returns a dictionary with unique id (formed by adding context_id and qas_id) as key and value as example with context paragraph, question and answer span (if is_training is true)"""
    id_example_dict = {}
    for paragraph_example in get_examples(input_json):
        question_examples = read_emr_entry(paragraph_example, is_training)
        for example in question_examples:
            uid = example.context_id + example.qas_id
            id_example_dict[uid] = example
    return id_example_dict


def read_emr_entry(entry, is_training):
    """ creates multiple training examples from given paragraph and question answer pairs from that paragraph """
    def is_whitespace(c):
        return c in " \t\r\n" or ord(c) == 0x202F

    examples = []
    note_ids = entry['note_id']  
    contexts = entry['context']
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in contexts:
        if (is_whitespace(c)):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    questions = []
    for i, qas in enumerate(entry["qas"]):  
        qas_id = qas['qas_id']
        question_text = qas['question']
        start_position = None
        end_position = None
        answer = None
        if is_training:
            answer = qas['answers']
            orig_answer_text = answer["text"]
            answer_offset = answer["offset"]
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length]
            # if answer is None or answer.offset
            if(contexts[answer_offset:answer_offset+answer_length].lower() != " ".join(doc_tokens[start_position:(end_position + 1)]).lower() or " ".join(doc_tokens[start_position:(end_position + 1)]).lower() != answer["text"].lower()):
                continue
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(answer["text"]))

        questions.append(question_text)
        example = EmrExample(
            context_id=int(note_ids),
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            answer=answer,
            start_position=start_position,
            end_position=end_position)
        examples.append(example)
    return examples


class EmrExample(object):
    """ returns an object of EmrExample with full context paragraph (without splitting the clinical notes in parts)"""
    def __init__(self, context_id, qas_id, question_text, doc_tokens, answer=None, start_position=None,
                 end_position=None):
        self.context_id = context_id
        self.qas_id = qas_id
        self.questions = question_text
        self.doc_tokens = doc_tokens
        self.answer = answer
        self.start_position = start_position
        self.end_position = end_position


class InputFeatures(object):
    """ return object of InputFeatures where is_impossible is True if there is no answer for the question in given context paragraph """
    def __init__(self,
                 unique_id,
                 context_id,
                 example_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=True):
        self.unique_id = unique_id
        self.context_index = context_id
        self.example_index = example_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def convert_single_emr_example(example, tokenizer, is_training):
    """ due to long length of clinical notes, for each EmrExample (context paragraph c , question q and answer span a), multiple training instances are created by 
    # doing a sliding window on clinical notes with stride of 384 length, to create new example with small context ci, above question q and answer span a (if answer 
    # lies in ci) or [cls]-[cls] (if answer is not present in ci)"""
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    features = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenize(tokenizer, token)
        tok_to_orig_index.extend([i] * len(sub_tokens))
        all_doc_tokens.extend(sub_tokens)

    query_tokens = []
    query_tokens.extend(tokenize(tokenizer, example.questions))
    if (len(query_tokens) > FLAGS.max_query_length):
        query_tokens = query_tokens[-FLAGS.max_query_length:]

    tok_start_position = 0
    tok_end_position = 0
    if (is_training):
        tok_start_position = orig_to_tok_index[example.start_position]
        if (example.end_position < len(example.doc_tokens) - 1):
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

    max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 3
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        length = min(length, max_tokens_for_doc)
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if (start_offset + length == len(all_doc_tokens)):
            break
        start_offset += min(length, FLAGS.doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids = []
        segment_ids.append(0)
        tokens.extend(query_tokens)
        segment_ids.extend([0] * len(query_tokens))
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                                  split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        assert len(tokens) == len(segment_ids)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < FLAGS.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == FLAGS.max_seq_length
        assert len(input_mask) == FLAGS.max_seq_length
        assert len(segment_ids) == FLAGS.max_seq_length
        start_position = None
        end_position = None
        is_impossible = False
        if is_training:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False

            if (not (tok_start_position >= doc_start and tok_end_position <= doc_end)):
                out_of_span = True
            if (out_of_span):
                start_position = 0
                end_position = 0
                is_impossible = True
            else:
                is_impossible = False
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
            answer_text = " ".join(tokens[start_position:(end_position + 1)])

        feature = InputFeatures(unique_id=-1, context_id=example.context_id, example_id=example.qas_id,
                                doc_span_index=doc_span_index,
                                tokens=tokens, token_to_orig_map=token_to_orig_map,
                                token_is_max_context=token_is_max_context,
                                input_ids=input_ids, input_mask=input_mask,
                                segment_ids=segment_ids, start_position=start_position,
                                end_position=end_position, is_impossible=is_impossible)

        features.append(feature)
    return features


def check_is_max_context(doc_spans, cur_span_index, position):
	"""Check if this is the 'max context' doc span for the token."""
	  # Because of the sliding window approach taken to scoring documents, a single
	  # token can appear in multiple documents. E.g.
	  #  Doc: the man went to the store and bought a gallon of milk
	  #  Span A: the man went to the
	  #  Span B: to the store and bought
	  #  Span C: and bought a gallon of
	  #  ...
	  #
	  # Now the word 'bought' will have two scores from spans B and C. We only
	  # want to consider the score with "maximum context", which we define as
	  # the *minimum* of its left and right context (the *sum* of left and
	  # right context will always be the same, of course).
	  #
	  # In the example the maximum context for 'bought' would be span C since
	  # it has 1 left context and 3 right context, while span B has 4 left context
	  # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class CreateTFExampleFnEmr(object):
    """ Creates tfrecords for training """
    def __init__(self, is_training):
        self.is_training = is_training
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    def process(self, example):
        emr_examples = read_emr_entry(example, self.is_training)
        input_features = []
        for emr_example in emr_examples:
            input_features.extend(
                convert_single_emr_example(emr_example, self.tokenizer, self.is_training))

        for input_feature in input_features:
            # input_feature.example_index = int(example["note_id"])
            input_feature.unique_id = (
                    input_feature.example_index + input_feature.doc_span_index + input_feature.context_index)

            def create_int_feature(values):
                return tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(values)))

            features = collections.OrderedDict()
            features["unique_ids"] = create_int_feature([input_feature.unique_id])
            features["context_indexes"] = create_int_feature([input_feature.context_index])
            features["doc_span_indexes"] = create_int_feature([input_feature.doc_span_index])
            features["example_indexes"] = create_int_feature([input_feature.example_index])
            features["input_ids"] = create_int_feature(input_feature.input_ids)
            features["input_mask"] = create_int_feature(input_feature.input_mask)
            features["segment_ids"] = create_int_feature(input_feature.segment_ids)

            if self.is_training:
                features["start_positions"] = create_int_feature(
                    [input_feature.start_position])
                features["end_positions"] = create_int_feature(
                    [input_feature.end_position])
            else:
                token_map = [-1] * len(input_feature.input_ids)
                for k, v in input_feature.token_to_orig_map.iteritems():
                    token_map[k] = v
                features["token_map"] = create_int_feature(token_map)

            yield tf.train.Example(features=tf.train.Features(
                feature=features)).SerializeToString()


def tokenize(tokenizer, text, apply_basic_tokenization=False):
	"""Tokenizes text, optionally looking up special tokens separately. Uses BERT's wordpeice tokenization"""
    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize
    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.vocab:
                tokens.append(token)
            else:
                tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))
    return tokens


def convert_emr_example_to_features(examples, tokenizer, is_training, output_fn):
	""" converts examples to features """
    num_spans_to_ids = collections.defaultdict(list)
    for example in examples:
        example_index = example.example_id
        features = convert_single_emr_example(example, tokenizer, is_training)
        num_spans_to_ids[len(features)].append(example.qas_id)

        for feature in features:
            feature.unique_id = feature.example_index + feature.context_index + feature.doc_span_index
            output_fn(feature)

    return num_spans_to_ids


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
	"""Creates a classification model to classify position of start and end token"""
    model = modeling.BertModel(config=bert_config, is_training=is_training, input_ids=input_ids,
                               input_mask=input_mask, token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)
    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable("cls/emr/output_weights", [2, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable("cls/emr/output_bias", [2], initializer=tf.zeros_initializer()) 

    # to be able to use checkpoints from SQUAD trained BERT QA model, replace 'emr' by 'squad' in names of above variables

    final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])

    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)
    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
    return (start_logits, end_logits)



def model_fn_builder(bert_config, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
	"""Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
    	"""The `model_fn` for TPUEstimator."""
        tf.logging.info("*******Features*******")
        for name in sorted(features.keys()):
            tf.logging.info("name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits) = create_model(bert_config = bert_config,
            is_training = is_training,
            input_ids = input_ids, 
            input_mask = input_mask,
            segment_ids= segment_ids,
            use_one_hot_embeddings = use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None

        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]
            def compute_loss(logits, positions):
            	""" computes the loss for predicted positions """
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss
            start_positions = features["start_positions"]
            end_positions = features["end_positions"]
            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)
            total_loss = (start_loss+end_loss)/2.0
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode = mode, loss = total_loss, train_op = train_op, scaffold_fn = scaffold_fn)
        else:
            raise ValueError("Only TRAIN mode is supported : %s" % (mode))

        return output_spec
    return model_fn


class EvalExample(object):
	"""Eval data available for a single example."""
    def __init__(self, example_id, example):
        # example_id in this case is unique id for each examples created by adding context_id and qas_id
        self.example_id = example_id
        self.example = example
        self.results = {}
        self.features = {}


def get_best_indexes(logits, n_best_size):
	"""Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


class ScoreSummary(object):
    def __init__(self):
        self.predicted_label = None
        self.short_span_score = None
        self.cls_token_score = None


def compute_predictions(example, max_answer_length):
	""" gives the predicted answer from model for one clinical note after looking into prediction across all doc-spans of clinical note"""
    predictions = []
    n_best_size = 10

    for unique_id, result in example.results.iteritems():
        if unique_id not in example.features:
            raise ValueError("No feature found with unique_id:", unique_id)
        token_map = example.features[unique_id]["token_map"].int64_list.value
        start_indexes = get_best_indexes(result["start_logits"], n_best_size)
        end_indexes = get_best_indexes(result["end_logits"], n_best_size)
        for start_index in start_indexes:
            for end_index in end_indexes:
                if end_index < start_index:
                    continue
                if token_map[start_index] == -1:
                    continue
                if token_map[end_index] == -1:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                summary = ScoreSummary()
                summary.short_span_score = (
                        result["start_logits"][start_index] +
                        result["end_logits"][end_index])
                summary.cls_token_score = (
                        result["start_logits"][0] + result["end_logits"][0])
                start_span = token_map[start_index]
                end_span = token_map[end_index] + 1

                score = summary.short_span_score - summary.cls_token_score
                predictions.append((score, summary, start_span, end_span))
    if(len(predictions)!=0):
        score, summary, start_span, end_span = sorted(predictions, reverse=True)[0]
    else:
        start_span = 0
        end_span = 0
    answer_span = Span(start_span, end_span)
    answer_tokens = example.example.doc_tokens[start_span:(end_span + 1)]
    answer_text = " ".join(answer_tokens)
    summary.predicted_label = {
        "example_id": example.example.qas_id,
        "context_id": example.example.context_id,
        "start_token": answer_span.start_token_idx,
        "end_token": answer_span.end_token_idx,
        "answer_text": answer_text
    }
    return summary


def compute_pred_dict(examples_dict, dev_features, raw_results,max_answer_length):
    """ computes the exact match and f1 score for predicted answers """
    raw_results_by_id = [(int(res["unique_id"] + 1), res) for res in raw_results]

    sess = tf.Session()
    all_candidates = examples_dict.items()
    example_ids = tf.to_int32(np.array([int(k) for k, _ in all_candidates])).eval(session=sess)
    examples_by_id = zip(example_ids, all_candidates)
    feature_ids = []
    features = []
    for f in dev_features:
        feature_ids.append(f.features.feature["unique_ids"].int64_list.value[0] + 1)
        features.append(f.features.feature)
    feature_ids = tf.to_int32(np.array(feature_ids)).eval(session=sess)
    features_by_id = zip(feature_ids, features)

    examples = []
    merged = sorted(examples_by_id + raw_results_by_id + features_by_id)
    for idx, datum in merged:
        if isinstance(datum, tuple):
            examples.append(EvalExample(datum[0], datum[1]))
        elif "token_map" in datum:
            examples[-1].features[idx] = datum
        else:
            examples[-1].results[idx] = datum

    tf.logging.info("Computing predictions...")
    summary_dict = {}
    emr_pred_dict = {}
    for e in examples:
        summary = compute_predictions(e,max_answer_length)
        summary_dict[e.example.qas_id + e.example.context_id] = summary
        emr_pred_dict[e.example.qas_id + e.example.context_id] = summary.predicted_label
        if len(summary_dict) % 100 == 0:
            tf.logging.info("Examples processed: %d", len(emr_pred_dict))
    tf.logging.info("Done computing predictions.")
    return emr_pred_dict


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
	"""Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
    	"""Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
    	"""The actual input function."""
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits"])


class FeatureWriter(object):
    """ writes the tfrecords for training and test data and if is_training true, it stores answer spans also in features """
    def __init__(self, filename, is_training):
        self._writer = tf.python_io.TFRecordWriter(filename)
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0

    def process_feature(self, feature):
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["context_indexes"] = create_int_feature([feature.context_index])
        features["doc_span_indexes"] = create_int_feature([feature.doc_span_index])
        features["example_indexes"] = create_int_feature([feature.example_index])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature([feature.start_position])
            features["end_positions"] = create_int_feature([feature.end_position])
        else:
            token_map = [-1] * len(feature.input_ids)
            for k, v in feature.token_to_orig_map.iteritems():
                token_map[k] = v
            features["token_map"] = create_int_feature(token_map)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()

Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])


def convert_examples_to_features(examples, tokenizer, is_training, output_fn):
    num_spans_to_ids = collections.defaultdict(list)

    for example in examples:
        # example_index = example.example_id
        features = convert_single_emr_example(example, tokenizer, is_training)
        num_spans_to_ids[len(features)].append(example.qas_id)

        for feature in features:
            # feature.example_index = example_index
            feature.unique_id = feature.example_index + feature.context_index + feature.doc_span_index
            output_fn(feature)

    return num_spans_to_ids


def validate_flags_or_throw(bert_config):
	"""Validate the input FLAGS or throw an exception."""
    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `{do_train,do_predict}` must be True.")
    if FLAGS.do_train:
        if not FLAGS.train_precomputed_file:
            raise ValueError("If `do_train` is True, then `train_precomputed_file` "
                             "must be specified.")
        if not FLAGS.train_num_precomputed:
            raise ValueError("If `do_train` is True, then `train_num_precomputed` "
                             "must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))

def normalize_answer(s):
	""" normalizes the input string """
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_gold_context, a_pred):
    """returns true if predicted answer lines in the window of +-20 character of ground truth answer
    # might have to fix this, this looks bit sloppy because prediction of words like "the" which probably says nothing about answer would make it to return true
    # work around could be that predict answer should have atleast 4 words and if they fall in vicinity of actual answer or it should match the exact answer """
    if(normalize_answer(a_pred) == normalize_answer(a_gold)):
        return True
    elif(len(normalize_answer(a_pred).split(' '))>2 and normalize_answer(a_pred) in normalize_answer(a_gold_context)):
        return True
    else:
        return False
        
def compute_f1(a_gold, a_pred):
	""" computes overlap between predicted and actual answer """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(dataset, predictions_dict):
	""" returns list of exact scores and f1 scores for all examples"""
    exact_scores = {}
    f1_scores = {}
    paragraph_list = dataset["paragraphs"]
    for paragraph in paragraph_list:
        context_id = int(paragraph["note_id"])
        for qas in paragraph['qas']:
            qas_id = int(qas["qas_id"])
            true_answer = normalize_answer(qas["answers"]["text"])
            predicted_answer = normalize_answer(predictions_dict[context_id+qas_id])
            true_answer_context = normalize_answer(qas["context_answer"])
            exact_scores[context_id+qas_id] = compute_exact(true_answer,true_answer_context, predicted_answer) 
            f1_scores[context_id+qas_id] = compute_f1(true_answer, predicted_answer)
    return exact_scores, f1_scores

def convert_to_dict(preds):
	""" returns dictionary of predicted answer and key as context_id + qas_id"""
    predictions_dict = {}
    predictions = preds["predictions"]
    for p in predictions:
        predictions_dict[int(p["example_id"])+int(p['context_id'])] = p["answer_text"]
    return predictions_dict

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
	

    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
            ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    validate_flags_or_throw(bert_config)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name, zone=FLAGS.tpu_zone,
                                                                              project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    save_checkpoints_steps = int(FLAGS.train_num_precomputed / FLAGS.train_batch_size)
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        keep_checkpoint_max=100,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        num_train_features = FLAGS.train_num_precomputed
        num_train_steps = int(int(num_train_features / FLAGS.train_batch_size) * FLAGS.num_train_epochs)

        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        model_dir=FLAGS.output_dir)
    if FLAGS.do_train:
        tf.logging.info("***** Running training on precomputed features *****")
        tf.logging.info("  Num split examples = %d", num_train_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_filename = FLAGS.train_precomputed_file
        train_input_fn = input_fn_builder(
            input_file=train_filename,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("train_precomputed_file")
    flags.mark_flag_as_required("train_num_precomputed")
    tf.app.run()
