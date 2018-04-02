from corpora_tools import *
from corpora_downloader import retrieve_cornell_corpora


sen_l1, sen_l2 = retrieve_cornell_corpora()
print("# Two consecutive sentences in a conversation")
print("Q:", sen_l1[0])
print("A:", sen_l2[0])

print("# Corpora length (i.e. number of sentences)")
print(len(sen_l1))
assert len(sen_l1) == len(sen_l2)


clean_sen_l1 = [clean_sentence(s) for s in sen_l1]
clean_sen_l2 = [clean_sentence(s) for s in sen_l2]

print("# Same sentence as before, but chunked and cleaned")
print("Q:", clean_sen_l1[0])
print("A:", clean_sen_l2[0])

filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1, clean_sen_l2)

print("# Filtered Corpora length (i.e. number of sentences)")
print(len(filt_clean_sen_l1))
assert len(filt_clean_sen_l1) == len(filt_clean_sen_l2)


dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=15000, storage_path="/tmp/l1_dict.p")
dict_l2 = create_indexed_dictionary(filt_clean_sen_l2, dict_size=15000, storage_path="/tmp/l2_dict.p")

idx_sentences_l1 = sentences_to_indexes(filt_clean_sen_l1, dict_l1)
idx_sentences_l2 = sentences_to_indexes(filt_clean_sen_l2, dict_l2)

print("# Same sentences as before, with their dictionary ID")
print("Q:", list(zip(filt_clean_sen_l1[0], idx_sentences_l1[0])))
print("A:", list(zip(filt_clean_sen_l2[0], idx_sentences_l2[0])))

max_length_l1 = extract_max_length(idx_sentences_l1)
max_length_l2 = extract_max_length(idx_sentences_l2)

print("# Max sentence sizes:")
print("Q:", max_length_l1)
print("A:", max_length_l2)


data_set = prepare_sentences(idx_sentences_l1, idx_sentences_l2, max_length_l1, max_length_l2)

print("# Prepared minibatch with paddings and extra stuff")
print("Q:", data_set[0][0])
print("A:", data_set[0][1])

print("# The sentence pass from X to Y tokens")
print("Q:", len(idx_sentences_l1[0]), "->", len(data_set[0][0]))
print("A:", len(idx_sentences_l2[0]), "->", len(data_set[0][1]))
