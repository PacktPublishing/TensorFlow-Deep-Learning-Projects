import time
import math
import sys
import pickle
import glob
import os
import tensorflow as tf

from seq2seq_model import Seq2SeqModel
from corpora_tools import *
from corpora_downloader import retrieve_cornell_corpora

path_l1_dict = "./chat/l1_dict.p"
path_l2_dict = "./chat/l2_dict.p"
model_dir = "./chat/chatbot_model"
model_checkpoints = model_dir + "/chatbot.ckpt"


def build_dataset(use_stored_dictionary=False):
    sen_l1, sen_l2 = retrieve_cornell_corpora(storage_path=".")
    clean_sen_l1 = [clean_sentence(s) for s in sen_l1][:30000] ### OTHERWISE IT DOES NOT RUN ON MY LAPTOP
    clean_sen_l2 = [clean_sentence(s) for s in sen_l2][:30000] ### OTHERWISE IT DOES NOT RUN ON MY LAPTOP
    filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1, clean_sen_l2, max_len=10)

    if not use_stored_dictionary:
        dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=10000, storage_path=path_l1_dict)
        dict_l2 = create_indexed_dictionary(filt_clean_sen_l2, dict_size=10000, storage_path=path_l2_dict)
    else:
        dict_l1 = pickle.load(open(path_l1_dict, "rb"))
        dict_l2 = pickle.load(open(path_l2_dict, "rb"))

    dict_l1_length = len(dict_l1)
    dict_l2_length = len(dict_l2)

    idx_sentences_l1 = sentences_to_indexes(filt_clean_sen_l1, dict_l1)
    idx_sentences_l2 = sentences_to_indexes(filt_clean_sen_l2, dict_l2)

    max_length_l1 = extract_max_length(idx_sentences_l1)
    max_length_l2 = extract_max_length(idx_sentences_l2)
    data_set = prepare_sentences(idx_sentences_l1, idx_sentences_l2, max_length_l1, max_length_l2)

    return (filt_clean_sen_l1, filt_clean_sen_l2), \
           data_set, \
           (max_length_l1, max_length_l2), \
           (dict_l1_length, dict_l2_length)


def cleanup_checkpoints(model_dir, model_checkpoints):
    for f in glob.glob(model_checkpoints + "*"):
        os.remove(f)
    try:
        os.mkdir(model_dir)
    except FileExistsError:
        pass


def get_seq2seq_model(session, forward_only, dict_lengths, max_sentence_lengths, model_dir):
    model = Seq2SeqModel(
        source_vocab_size=dict_lengths[0],
        target_vocab_size=dict_lengths[1],
        buckets=[max_sentence_lengths],
        size=256,
        num_layers=2,
        max_gradient_norm=5.0,
        batch_size=128,
        learning_rate=1.0,
        learning_rate_decay_factor=0.99,
        forward_only=forward_only,
        dtype=tf.float16)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    with tf.Session() as sess:
        model = get_seq2seq_model(sess, False, dict_lengths, max_sentence_lengths, model_dir)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        bucket = 0
        steps_per_checkpoint = 100
        max_steps = 20000

        while current_step < max_steps:

            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch([data_set], bucket)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket, False)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1

            if current_step % steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step {} learning rate {} step-time {} perplexity {}".format(
                       model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

                sess.run(model.learning_rate_decay_op)

                model.saver.save(sess, model_checkpoints, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                encoder_inputs, decoder_inputs, target_weights = model.get_batch([data_set], bucket)
                _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket, True)
                eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                print("  eval: perplexity {}".format(eval_ppx))
                sys.stdout.flush()


if __name__ == "__main__":
    _, data_set, max_sentence_lengths, dict_lengths = build_dataset(False)
    cleanup_checkpoints(model_dir, model_checkpoints)
    train()

