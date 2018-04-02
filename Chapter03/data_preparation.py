import os
annotation_dir = 'Flickr8k_text'


def read_file(file_name):
    with open(os.path.join(annotation_dir, file_name), 'rb') as file_handle:
        file_lines = file_handle.read().splitlines()
    return file_lines


train_image_paths = read_file('Flickr_8k.trainImages.txt')
test_image_paths = read_file('Flickr_8k.testImages.txt')
captions = read_file('Flickr8k.token.txt')

print(len(train_image_paths))
print(len(test_image_paths))
print(len(captions))

def get_vocab():
    image_caption_map = {}
    unique_words = set()
    max_words = 0
    for caption in captions:
        caption = caption.decode("utf-8")
        image_name = caption.split('#')[0]

        image_caption = caption.split('#')[1].split('\t')[1]
        if image_name not in image_caption_map.keys():
            image_caption_map[image_name] = [image_caption]
        else:
            image_caption_map[image_name].append(image_caption)
        caption_words = image_caption.split()
        max_words = max(max_words, len(caption_words))
        [unique_words.add(caption_word) for caption_word in caption_words]
    unique_words = list(unique_words)
    word_to_index_map = {}
    index_to_word_map = {}
    for index, unique_word in enumerate(unique_words):
        word_to_index_map[unique_word] = index
        index_to_word_map[index] = unique_word
    return image_caption_map, max_words, unique_words, word_to_index_map, index_to_word_map



