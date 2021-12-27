import pandas as pd
import re
import numpy as np
from scipy.spatial import distance
from transformers import AutoTokenizer, AutoModel
import torch
import pickle


# constrains
data_path = r"data\dataset_lubimyczytac.csv"
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')


def process_book_description(sample):
    sample = sample.lower()
    sample = re.sub('[^\w\d\s]+', ' ', sample)
    sample = re.sub(' +', ' ', sample)
    return sample


def get_sample_embedding(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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


if __name__ == '__main__':
    df = pd.read_csv(data_path, encoding='utf-8', sep=';')
    df = df.fillna('')
    df['sample'] = df['title'] + ' ' + df['author'] + ' ' + df['genre'] + ' ' + df['tags'] + ' ' + df['description']
    df['label'] = df['genre'] + ' ' + df['tags'] + ' ' + df['title']
    df['sample'] = df['sample'].apply(process_book_description)
    df['length'] = df['sample'].apply(lambda x: len(x.split(' ')))
    df['sample'] = df['sample'].apply(lambda x: ' '.join(x.split(' ')))
    data = dict(zip(df['idx'].values, df['sample'].values))

    sentence_embeddings = np.zeros(shape=(len(df.index), int(768/2)))

    for idx, text in data.items():
        # Tokenize sentences
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        output = get_sample_embedding(model_output, encoded_input['attention_mask'])
        output = np.array(output)

        m = 2
        output = output.reshape(-1, m).mean(axis=1)

        sentence_embeddings[idx] = output

    vectors = calculate_embedding_distances(sentence_embeddings)
    save_file(vectors, filename='results_transformer.pkl')
