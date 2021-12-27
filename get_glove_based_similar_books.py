from gensim.models import KeyedVectors
import pandas as pd
import re
import pickle
import numpy as np
from scipy.spatial import distance


# constrains
data_path = r"dataset.csv"
word2vec = KeyedVectors.load_word2vec_format(r"glove\glove_100_3_polish.txt")


def process_book_description(sample):
    sample = sample.lower()
    sample = re.sub('[^\w\d\s]+', ' ', sample)
    sample = re.sub(' +', ' ', sample)
    return sample


def calculate_embedding_distances(sentence_embeddings):
    vectors = {}
    for i in range(sentence_embeddings.shape[0]):
        distances_val = distance.cdist([sentence_embeddings[i]], sentence_embeddings, "cosine")[0]
        distances = distances_val.argsort()[1:11]
        vectors[idx] = distances
    return vectors


def save_file(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_sentence_embedding(sample, model):
    sample = sample.split(' ')
    embeddings = []
    for i in range(len(sample)):
        try:
            embeddings.append(model[sample[i]])
        except Exception as e:
            print(e)

    return np.mean(embeddings, axis=0)


if __name__ == '__main__':

    df = pd.read_csv(data_path, encoding='utf-8', sep=';')
    df = df.fillna('')
    df['sample'] = df['title'] + ' ' + df['author'] + ' ' + df['genre'] + ' ' + df['tags']
    df['sample'] = df['sample'].apply(process_book_description)
    texts = df['sample'].tolist()
    labels = df.idx.tolist()

    books = dict(zip(labels, texts))

    sentence_embeddings = np.zeros(shape=(len(texts), 100))

    for idx, sample in books.items():
        sentence_embeddings[idx] = get_sentence_embedding(sample, word2vec)

    vectors = calculate_embedding_distances(sentence_embeddings)
    save_file(vectors, filename='results_glove.pkl')
