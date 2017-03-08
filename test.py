## Checkpoint
import json
import os
import logging
import pdb

import tensorflow as tf
# from train import FLAGS
import nlc_data
import numpy as np
from train import create_model, FLAGS
from util import pair_iter

# with open(os.path.join("data", "flags.json"), 'r') as fout:
#     FLAGS = json.load(fout)

best_epoch = 1
vocab_size = 42
checkpoint_path = os.path.join(FLAGS.train_dir, "best.ckpt")
with tf.Session() as sess:
    logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, vocab_size, False)

    model.saver.restore(sess, checkpoint_path + ("-%d" % best_epoch))

    valid_costs, valid_lengths = [], []
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter("data/char/valid.ids.x", "data/char/valid.ids.y", 1,
                                                                            FLAGS.num_layers):
        # cost = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
        # valid_costs.append(cost * target_mask.shape[1])
        # valid_lengths.append(np.sum(target_mask[1:, :]))
        enc = model.encode(sess, source_tokens, source_mask) # (48, 128, 256)
        # print(enc.shape)
        # pdb.set_trace()
        dec = model.decode(sess, enc, target_tokens, target_mask) # (50, 128, 42)
        # print(dec)
        print(source_tokens)
        print(target_tokens)
        print(np.argmax(dec[0], axis=2))
        pdb.set_trace()