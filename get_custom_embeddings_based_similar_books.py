from keras.models import model_from_json
from os.path import join
import numpy as np
import pickle
import pandas as pd
from scipy.spatial import distance

# constrains
path = r"dataset.csv"


def load_model():
    with open('model.json') as file:
        model = model_from_json(file.read())
    model.load_weights('weights.h5')
    return model


def get_book_embeddings(model):
    book_embedding_layer = model.get_layer('book_embedding')
    weights = book_embedding_layer.get_weights()[0]
    weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
    np.sum(np.square(weights[0]))
    return weights


def calculate_embedding_distances(sentence_embeddings):
    vectors = {}
    for i in range(sentence_embeddings.shape[0]):
        distances_val = distance.cdist([sentence_embeddings[i]], sentence_embeddings, "cosine")[0]
        distances = distances_val.argsort()[1:11]
        vectors[i] = distances
    return vectors


def save_file(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    model = load_model()
    embeddings = get_book_embeddings(model)
    vectors = calculate_embedding_distances(embeddings)
    save_file(vectors, filename='results_custom.pkl')
