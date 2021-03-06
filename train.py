# Copyright 2016 Stanford University
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf

import nlc_data
import nlc_model
from util import get_tokenizer
from util import pair_iter

logging.basicConfig(level=logging.INFO)

FLAGS = {
  "learning_rate": 0.0003, #"Learning rate.")
  "learning_rate_decay_factor": 0.95,# "Learning rate decays by this much.")
  "max_gradient_norm": 10.0,# "Clip gradients to this norm.")
  "dropout": 0.15,# "Fraction of units randomly dropped on non-recurrent connections.")
  "batch_size": 64, #"Batch size to use during training.")
  "epochs": 40,# "Number of epochs to train.")
  "size": 256,# "Size of each model layer.")
  "num_layers": 2, #"Number of layers in the model.")
  "max_vocab_size": 42,# "Vocabulary size limit.")
  "max_seq_len": 50, #"Maximum sequence length.")
  "data_dir": "data", #"Data directory")
  "output_dir": "data", #"Training directory.")
  "tokenizer": "CHAR", #"BPE / CHAR / WORD.")
  "optimizer": "adam", #"adam / sgd")
  "print_every": 100 #"How many iterations to do per print.")
}


def create_model(session, vocab_size, forward_only):
    model = nlc_model.NLCModel(
        vocab_size, FLAGS['size'], FLAGS['num_layers'], FLAGS['max_gradient_norm'], FLAGS['batch_size'],
        FLAGS['learning_rate'], FLAGS['learning_rate_decay_factor'], FLAGS['dropout'],
        forward_only=forward_only, optimizer=FLAGS['optimizer'])
    ckpt = tf.train.get_checkpoint_state(FLAGS['output_dir'])
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def validate(model, sess, x_dev, y_dev):
    valid_costs, valid_lengths = [], []
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev, y_dev, FLAGS['batch_size'],
                                                                            FLAGS['num_layers'], FLAGS):
        cost = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
        valid_costs.append(cost * target_mask.shape[1])
        valid_lengths.append(np.sum(target_mask[1:, :]))
    valid_cost = sum(valid_costs) / float(sum(valid_lengths))
    return valid_cost


def train():
    """Train a translation model using NLC data."""
    # Prepare NLC data.
    logging.info("Preparing NLC data in %s" % FLAGS['data_dir'])

    x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
        FLAGS['data_dir'] + '/' + FLAGS['tokenizer'].lower(), FLAGS['max_vocab_size'],
        tokenizer=get_tokenizer(FLAGS))
    vocab, _ = nlc_data.initialize_vocabulary(vocab_path)
    vocab_size = len(vocab)
    logging.info("Vocabulary size: %d" % vocab_size)

    if not os.path.exists(FLAGS['output_dir']):
        os.makedirs(FLAGS['output_dir'])
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS['output_dir']))
    logging.getLogger().addHandler(file_handler)

    print(FLAGS)
    with open(os.path.join(FLAGS['output_dir'], "flags.json"), 'w') as fout:
        json.dump(FLAGS, fout)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        logging.info("Creating %d layers of %d units." % (FLAGS['num_layers'], FLAGS['size']))
        model = create_model(sess, vocab_size, False)

        #logging.info('Initial validation cost: %f' % validate(model, sess, x_dev, y_dev))
        logging.info('Initial validation cost')

        if False:
            tic = time.time()
            params = tf.trainable_variables()
            num_params = sum(list(map(lambda t: np.prod(tf.shape(t.value()).eval()), params)))
            toc = time.time()
            print("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        epoch = 0
        best_epoch = 0
        previous_losses = []
        exp_cost = None
        exp_length = None
        exp_norm = None
        total_iters = 0
        start_time = time.time()
        while (FLAGS['epochs'] == 0 or epoch < FLAGS['epochs']):
            epoch += 1
            current_step = 0

            ## Train
            epoch_tic = time.time()
            for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_train, y_train, FLAGS['batch_size'],
                                                                                    FLAGS['num_layers'], FLAGS):
                # Get a batch and make a step.
                tic = time.time()

                grad_norm, cost, param_norm = model.train(sess, source_tokens, source_mask, target_tokens, target_mask)

                toc = time.time()
                iter_time = toc - tic
                total_iters += np.sum(target_mask)
                tps = total_iters / (time.time() - start_time)
                current_step += 1

                lengths = np.sum(target_mask, axis=0)
                mean_length = np.mean(lengths)
                std_length = np.std(lengths)

                if not exp_cost:
                    exp_cost = cost
                    exp_length = mean_length
                    exp_norm = grad_norm
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * cost
                    exp_length = 0.99 * exp_length + 0.01 * mean_length
                    exp_norm = 0.99 * exp_norm + 0.01 * grad_norm

                cost = cost / mean_length

                if current_step % FLAGS['print_every'] == 0:
                    logging.info(
                        'epoch %d, iter %d, cost %f, exp_cost %f, grad norm %f, param norm %f, tps %f, length mean/std %f/%f' %
                        (epoch, current_step, cost, exp_cost / exp_length, grad_norm, param_norm, tps, mean_length,
                         std_length))

                if current_step % (FLAGS['print_every'] * 10) == 0:
                    ## Validate
                    epoch_toc = time.time()
                    valid_cost = validate(model, sess, x_dev, y_dev)
                    logging.info("Epoch %d Validation cost: %f time: %f" % (epoch, valid_cost, epoch_toc - epoch_tic))
                    ## Checkpoint
                    checkpoint_path = os.path.join(FLAGS['output_dir'], "best.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=epoch)

            if len(previous_losses) > 2 and valid_cost > previous_losses[-1]:
                logging.info("Annealing learning rate by %f" % FLAGS['learning_rate_decay_factor'])
                sess.run(model.learning_rate_decay_op)
                model.saver.restore(sess, checkpoint_path + ("-%d" % best_epoch))
            else:
                previous_losses.append(valid_cost)
                best_epoch = epoch
                model.saver.save(sess, checkpoint_path, global_step=epoch)
            sys.stdout.flush()


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
