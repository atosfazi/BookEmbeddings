import pandas as pd
import re
import random
import numpy as np
from keras.layers import Input, Embedding, Dot, Reshape
from keras.models import Model
from unidecode import unidecode


# constrains
path = r"dataset.csv"
positive_samples_per_batch = 512


def convert_tags_to_idx(tags):
    splitted = tags.split(' ')
    splitted = [tag_idx[x] for x in splitted if x in tag_idx]
    return splitted


def book_embedding_model(embedding_size=50):
    book = Input(name='book', shape=[1])
    tag = Input(name='tag', shape=[1])

    book_embedding = Embedding(name='book_embedding',
                               input_dim=len(idx_title),
                               output_dim=embedding_size)(book)

    tag_embedding = Embedding(name='tag_embedding',
                               input_dim=len(idx_tag),
                               output_dim=embedding_size)(tag)

    merged = Dot(name='dot_product', normalize=True, axes=2)([book_embedding, tag_embedding])
    merged = Reshape(target_shape=[1])(merged)

    model = Model(inputs=[book, tag], outputs=merged)
    model.compile(optimizer='Adam', loss='mse')
    return model


def generator(pairs, n_positive=50):
    """Generate batches of samples for training"""
    batch_size = n_positive * 2
    batch = np.zeros((batch_size, 3))

    neg_label = -1

    while True:
        for idx, (book_id, tag_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (book_id, tag_id, 1)

        idx += 1

        while idx < batch_size:
            random_book = random.randrange(len(book_idx))
            random_tag = random.randrange(len(tag_idx))

            if (random_book, random_tag) not in pairs:
                batch[idx, :] = (random_book, random_tag, neg_label)
                idx += 1

        np.random.shuffle(batch)
        yield {'book': batch[:, 0], 'tag': batch[:, 1]}, batch[:, 2]


def save_model(model):
    with open('model.json', 'w') as f:
        f.write(model.to_json())
    model.save('weights.h5')


if __name__ == '__main__':

    # data preprocessing
    df = pd.read_csv(path, encoding='utf-8-sig', sep=';')
    df['tags'] = df['tags'] + ' ' + df['genre']
    df['tags'] = df.tags.astype(str)
    df['tags'] = df.tags.apply(lambda x: x.lower())
    df['tags'] = df.tags.apply(lambda x: re.sub(r"[^a-z0-9]+", ' ', unidecode(x.lower())))
    tags = list(set(df.tags.values))
    tags = [x for x in tags if not x.startswith('http')]
    tags = [x for x in tags if not x.startswith('www')]

    tags_final = []
    for tag in tags:
        splitted = tag.split(' ')
        for x in splitted:
            if len(x) > 2:
                tags_final.append(x)

    # get unique tags dict
    tags_final = list(set(tags_final))
    idx_tag = dict(enumerate(tags_final))
    tag_idx = {v: k for k, v in idx_tag.items()}

    # get books dict
    idx_book = dict(zip(df.idx, df.title))
    book_idx = {v: k for k, v in idx_book.items()}

    # get tain data
    df = df[['idx', 'title', 'tags']]
    df['tags'] = df['tags'].apply(convert_tags_to_idx)
    df = df.explode('tags')
    train_pairs = list(zip(df.idx, df.tags))

    # training
    model = book_embedding_model()

    gen = generator(train_pairs, positive_samples_per_batch)

    h = model.fit(gen, epochs=12,
                  steps_per_epoch=len(train_pairs) // positive_samples_per_batch,
                  verbose=2)

    save_model(model)
