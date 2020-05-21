# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Converts an NQ dataset file to tf examples.
Notes:
  this script should be run to generate tf records for training data
  using the json file of training data. It prints the number of examples
  at the end which is further passed as argument for num_precomputed_train,
  while running training code - run_emr_tpu / run_emr_tpu_has_answer

  Set is_training true for evaluation and training tfrecords. For prediction, set it to False.
"""

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

import gzip
import random
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 384,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_string(
    "input_json", None,
    "Gzipped files containing emr examples in Json format")

flags.DEFINE_string("output_tfrecord", None,
                    "Output tf record file with all features extracted.")

flags.DEFINE_bool(
    "is_training", True,
    "Whether to prepare features for training or for evaluation. Eval features "
    "don't include gold labels, but include wordpiece to html token maps.")

flags.DEFINE_integer(
    "max_examples", 0,
    "If positive, stop once these many examples have been converted.")

_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)

def get_examples(input_json):
  """ reads the train/test/validation json to give the examples, where each example is a paragraph with multiple question-answer instances """
  with open(input_json,"r") as f:
    data = json.load(f)
    for j in range(len(data['paragraphs'])):
      yield data['paragraphs'][j]


def read_emr_entry(entry, is_training):
  """ reads input json to return list of examples where each example has context paragraph, question and answer span(if is_training is true)"""
  def is_whitespace(c):
    return c in " \t\r\n" or ord(c) == 0x202F

  examples = []
  note_ids = entry['note_id'] # context_id for NQ dataset
  contexts = entry['context']
  doc_tokens = []
  char_to_word_offset = []
  prev_is_whitespace = True
  for c in contexts:
    if(is_whitespace(c)):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    char_to_word_offset.append(len(doc_tokens)-1)

  questions = []
  for i, qas in enumerate(entry["qas"]): # qas is equivalent to questions in NQ dataset
    qas_id = qas['qas_id']
    question_text = qas['question']
    start_position = None
    end_position = None
    answer = None
    if is_training:
      answer= qas['answers']
      orig_answer_text = answer["text"]
      answer_offset = answer["offset"]
      answer_length = len(orig_answer_text)
      start_position = char_to_word_offset[answer_offset]
      end_position = char_to_word_offset[answer_offset + answer_length-1]
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
  """ class for emr example which has context, question, respective ids and start and end position of answer in context """
  def __init__(self, context_id, qas_id, question_text, doc_tokens, answer = None, start_position = None, end_position = None):
    self.context_id = context_id
    self.qas_id = qas_id
    self.questions = question_text
    self.doc_tokens = doc_tokens
    self.answer = answer
    self.start_position = start_position
    self.end_position = end_position

class InputFeatures(object):
  """A single set of features of data."""
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
               has_answer = 0):
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
    self.has_answer = has_answer



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
    tok_to_orig_index.extend([i]*len(sub_tokens))
    all_doc_tokens.extend(sub_tokens)

  query_tokens = []
  query_tokens.extend(tokenize(tokenizer, example.questions))
  if(len(query_tokens) > FLAGS.max_query_length):
    query_tokens = query_tokens[-FLAGS.max_query_length:]

  tok_start_position = 0
  tok_end_position =0
  if(is_training):
    tok_start_position = orig_to_tok_index[example.start_position]
    if(example.end_position < len(example.doc_tokens) -1):
      tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    else :
      tok_end_position = len(all_doc_tokens)-1

  max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 3
  _DocSpan = collections.namedtuple("DocSpan",["start","length"])
  doc_spans = []
  start_offset = 0
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    length = min(length, max_tokens_for_doc)
    doc_spans.append(_DocSpan(start = start_offset, length = length))
    if(start_offset + length == len(all_doc_tokens)):
      break
    start_offset += min(length, FLAGS.doc_stride)

  for(doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids = []
    segment_ids.append(0)
    tokens.extend(query_tokens)
    segment_ids.extend([0]*len(query_tokens))
    tokens.append("[SEP]")
    segment_ids.append(0)

    # replace xrange with range if running in python3 as it doesnt support xrange
    for i in xrange(doc_span.length):
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
    input_mask = [1]*len(input_ids)

    while len(input_ids) < FLAGS.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids)  == FLAGS.max_seq_length
    start_position = None
    end_position = None
    if is_training:
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      out_of_span = False

      if(not(tok_start_position >= doc_start and tok_end_position<=doc_end)):
        out_of_span = True
      if(out_of_span):
        start_position = 0
        end_position = 0
        has_answer = 0
      else :
        has_answer = 1
        doc_offset = len(query_tokens)+2
        start_position = tok_start_position - doc_start + doc_offset
        end_position = tok_end_position - doc_start + doc_offset
      answer_text = " ".join(tokens[start_position:(end_position + 1)])

    feature = InputFeatures(unique_id = -1, context_id = example.context_id, example_id = example.qas_id, doc_span_index =doc_span_index ,
     tokens = tokens ,token_to_orig_map = token_to_orig_map , token_is_max_context= token_is_max_context,
      input_ids = input_ids , input_mask= input_mask,
      segment_ids = segment_ids, start_position = start_position ,
      end_position = end_position, has_answer = has_answer)

    features.append(feature)

  return features

def check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""
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
  """Functor for creating NQ tf.Examples."""
  def __init__(self, is_training):
    self.is_training = is_training
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  def process(self, example):
    """Coverts an NQ example in a list of serialized tf examples."""
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
        features["has_answer"] = create_int_feature(
            [input_feature.has_answer])
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


def main(_):
  examples_processed = 0
  num_examples_with_correct_context = 0
  creator_fn = CreateTFExampleFnEmr(is_training=FLAGS.is_training)

  instances = []
  for example in get_examples(FLAGS.input_json):
    for instance in creator_fn.process(example):
      instances.append(instance)
    if examples_processed % 100 == 0:
      tf.logging.info("Examples processed: %d", examples_processed)
    examples_processed += 1
    if FLAGS.max_examples > 0 and examples_processed >= FLAGS.max_examples:
      break
  random.shuffle(instances)
  tf.logging.info("Total number of training examples : %d", len(instances))
  with tf.python_io.TFRecordWriter(FLAGS.output_tfrecord) as writer:
    for instance in instances:
      writer.write(instance)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("input_json")
  flags.mark_flag_as_required("output_tfrecord")
  tf.app.run()
