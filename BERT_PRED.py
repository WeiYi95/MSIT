#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author: Wei Yi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
#from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.python.ops import math_ops
import tf_metrics
import pickle
import utils


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "training_examples3", 10000,
    "Total training examples",
)

flags.DEFINE_integer(
    "eval_examples33", 1000,
    "Total eval examples",
)

flags.DEFINE_integer(
    "pred_examples3", 1000,
    "Total pred examples",
)

flags.DEFINE_string(
    "data_dir3", "./",
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file3", "./bert_config.json",
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name3", "SEG", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir3", "temp",
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint3", "output/model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case3", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length3", 24,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train3", False,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu3", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval3", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict3", True, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size3", 128, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size3", 128, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size3", 64, "Total batch size for predict.")

flags.DEFINE_float("learning_rate3", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs3", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion3", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps3", 100000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop3", 50,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file3", "./vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master3", None, "[Optional] TensorFlow master3 URL.")
flags.DEFINE_integer(
    "num_tpu_cores3", 8,
    "Only used if `use_tpu3` is True. Total number of TPU cores to use.")


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length3, 5])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities, axis=-1)
        return (loss, per_example_loss, logits, predict)
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
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
            #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids, predictions, 5, [1, 2], average="macro")
                #recall = tf_metrics.recall(label_ids, predictions, 13, [1, 2, 4, 5, 6, 7, 8, 9], average="macro")
                #f = tf_metrics.f1(label_ids, predictions, 13, [1, 2, 4, 5, 6, 7, 8, 9], average="macro")
                #
                return {
                    "eval_precision": precision
                    #"eval_recall": recall,
                    #"eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn

def get_labels():
    #the X is for english token
    #return ["[BOS]", "[IOS]", "X", "[CLS]", "[SEP]"]
    return ["[BOS]", "[IOS]", "[CLS]", "[SEP]"]

def BERT_PRED():
    tf.logging.set_verbosity(tf.logging.INFO)

    #if not FLAGS.do_train and not FLAGS.do_eval3:
    #    raise ValueError("At least one of `do_train` or `do_eval3` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file3)

    if FLAGS.max_seq_length3 > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length3, bert_config.max_position_embeddings))

    task_name3 = FLAGS.task_name3.lower()

    label_list = get_labels()

    tpu_cluster_resolver = None
    if FLAGS.use_tpu3 and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master3,
        model_dir=FLAGS.output_dir3,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps3,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop3,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint3,
        learning_rate=FLAGS.learning_rate3,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu3,
        use_one_hot_embeddings=FLAGS.use_tpu3)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu3,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size3,
        eval_batch_size=FLAGS.eval_batch_size3,
        predict_batch_size=FLAGS.predict_batch_size3)

    if FLAGS.do_predict3:
        token_path = os.path.join(FLAGS.output_dir3, "token_test.txt")
        with open('./label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_file = utils.read_from_pkl("output_filename.pkl")
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", FLAGS.pred_examples3)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size3)
        if FLAGS.use_tpu3:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu3 else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length3,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir3, "pred.txt")
        with open(output_predict_file, 'w') as writer:
            for prediction in result:
                output_line = " ".join(id2label[id] for id in prediction if id != 0) + "\n"
                writer.write(output_line)
