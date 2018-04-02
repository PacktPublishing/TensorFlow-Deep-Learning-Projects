import re

def read_conversations(storage_path, storage_dir):
    filename = storage_path + "/" + storage_dir + "/cornell movie-dialogs corpus/movie_conversations.txt"
    with open(filename, "r", encoding="ISO-8859-1") as fh:
        conversations_chunks = [line.split(" +++$+++ ") for line in fh]
    return [re.sub('[\[\]\']', '', el[3].strip()).split(", ") for el in conversations_chunks]


def read_lines(storage_path, storage_dir):
    filename = storage_path + "/" + storage_dir + "/cornell movie-dialogs corpus/movie_lines.txt"
    with open(filename, "r", encoding="ISO-8859-1") as fh:
        lines_chunks = [line.split(" +++$+++ ") for line in fh]
    return {line[0]: line[-1].strip() for line in lines_chunks}


def get_tokenized_sequencial_sentences(list_of_lines, line_text):
    for line in list_of_lines:
        for i in range(len(line) - 1):
            yield (line_text[line[i]].split(" "), line_text[line[i+1]].split(" "))

def download_and_decompress(url, storage_path, storage_dir):
    import os.path

    directory = storage_path + "/" + storage_dir
    zip_file = directory + ".zip"
    a_file = directory + "/cornell movie-dialogs corpus/README.txt"

    if not os.path.isfile(a_file):
        import urllib.request
        import zipfile

        urllib.request.urlretrieve(url, zip_file)

        with zipfile.ZipFile(zip_file, "r") as zfh:
            zfh.extractall(directory)
    return

def retrieve_cornell_corpora(storage_path="/tmp", storage_dir="cornell_movie_dialogs_corpus"):
    download_and_decompress("http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
                            storage_path,
                            storage_dir)

    conversations = read_conversations(storage_path, storage_dir)
    lines = read_lines(storage_path, storage_dir)

    return tuple(zip(*list(get_tokenized_sequencial_sentences(conversations, lines))))
