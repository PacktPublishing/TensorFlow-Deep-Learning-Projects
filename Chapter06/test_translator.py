import pickle
import sys
import numpy as np
import tensorflow as tf

import data_utils
from train_translator import (get_seq2seq_model, path_l1_dict, path_l2_dict,
                              build_dataset)


model_dir = "/tmp/translate"


def decode():
  with tf.Session() as sess:
    model = get_seq2seq_model(sess, True, dict_lengths, max_sentence_lengths, model_dir)
    model.batch_size = 1
    bucket = 0

    for idx in range(len(data_set))[:5]:
        print("-------------------")
        print("Source sentence: ", sentences[0][idx])
        print("Source tokens: ", data_set[idx][0])
        print("Ideal tokens out: ", data_set[idx][1])
        print("Ideal sentence out: ", sentences[1][idx])

        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket: [(data_set[idx][0], [])]}, bucket)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if data_utils.EOS_ID in outputs:
            outputs = outputs[1:outputs.index(data_utils.EOS_ID)]

        print("Model output: ",  " ".join([tf.compat.as_str(inv_dict_l2[output]) for output in outputs]))
        sys.stdout.flush()



if __name__ == "__main__":
    dict_l2 = pickle.load(open(path_l2_dict, "rb"))
    inv_dict_l2 = {v: k for k, v in dict_l2.items()}

    build_dataset(True)
    sentences, data_set, max_sentence_lengths, dict_lengths = build_dataset(False)
    try:
        print("Reading from", model_dir)
        print("Dictionary lengths", dict_lengths)
        print("Bucket size", max_sentence_lengths)
    except NameError:
        print("One or more variables not in scope. Translation not possible")
        exit(-1)

    decode()
