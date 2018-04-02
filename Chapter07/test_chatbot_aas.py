import pickle
import sys
import numpy as np
import tensorflow as tf

import data_utils
from corpora_tools import clean_sentence, sentences_to_indexes, prepare_sentences
from train_chatbot import get_seq2seq_model, path_l1_dict, path_l2_dict

model_dir = "./chat/chatbot_model"


def prepare_sentence(sentence, dict_l1, max_length):
    sents = [sentence.split(" ")]
    clean_sen_l1 = [clean_sentence(s) for s in sents]
    idx_sentences_l1 = sentences_to_indexes(clean_sen_l1, dict_l1)
    data_set = prepare_sentences(idx_sentences_l1, [[]], max_length, max_length)
    sentences = (clean_sen_l1, [[]])
    return sentences, data_set


def decode(data_set):
  with tf.Session() as sess:
    model = get_seq2seq_model(sess, True, dict_lengths, max_sentence_lengths, model_dir)
    model.batch_size = 1
    bucket = 0

    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
      {bucket: [(data_set[0][0], [])]}, bucket)
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket, True)
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    if data_utils.EOS_ID in outputs:
        outputs = outputs[1:outputs.index(data_utils.EOS_ID)]

  tf.reset_default_graph()
  return " ".join([tf.compat.as_str(inv_dict_l2[output]) for output in outputs])


if __name__ == "__main__":
    dict_l1 = pickle.load(open(path_l1_dict, "rb"))
    dict_l1_length = len(dict_l1)

    dict_l2 = pickle.load(open(path_l2_dict, "rb"))
    dict_l2_length = len(dict_l2)
    inv_dict_l2 = {v: k for k, v in dict_l2.items()}

    max_lengths = 10
    dict_lengths = (dict_l1_length, dict_l2_length)
    max_sentence_lengths = (max_lengths, max_lengths)


    from bottle import route, run, request
    @route('/api')
    def api():
        in_sentence = request.query.sentence
        _, data_set = prepare_sentence(in_sentence, dict_l1, max_lengths)
        resp = [{"in": in_sentence, "out": decode(data_set)}]
        return dict(data=resp)


    run(host='127.0.0.1', port=8080, reloader=True, debug=True)
