from data_preparation import get_vocab

from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence

image_caption_map, max_words, unique_words, \
word_to_index_map, index_to_word_map = get_vocab()
vocabulary_size = len(unique_words)


def data_generator():
    return None


image_model = Sequential()
image_model.add(Dense(128, input_dim=4096, activation='relu'))

image_model.add(RepeatVector(max_words))

lang_model = Sequential()
lang_model.add(Embedding(vocabulary_size, 256, input_length=max_words))
lang_model.add(LSTM(256, return_sequences=True))
lang_model.add(TimeDistributed(Dense(128)))

model = Sequential()
model.add(Merge([image_model, lang_model], mode='concat'))
model.add(LSTM(1000, return_sequences=False))
model.add(Dense(vocabulary_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
batch_size = 32
epochs = 10
total_samples = 9
model.fit_generator(data_generator(batch_size=batch_size), steps_per_epoch=total_samples / batch_size,
                    epochs=epochs, verbose=2)

