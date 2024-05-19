import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

train_df = pd.read_csv('train.tsv', sep="\t", header=None, names=["label", "text"], on_bad_lines='skip')
train_df['text'].fillna('', inplace=True)
dev_df = pd.read_csv('dev-0/in.tsv', sep="\t", header=None, names=["text"], on_bad_lines='skip')
dev_df['text'].fillna('', inplace=True)
dev_df['label'] = 0
test_df = pd.read_csv('test-A/in.tsv', sep="\t", header=None, names=["text"], on_bad_lines='skip')
test_df['text'].fillna('', inplace=True)
test_df['label'] = 0

w2v_model = KeyedVectors.load("word2vec/word2vec_100_3_polish.bin")

def text_to_vector(text, model_w2v):
    if not isinstance(text, str):
        return np.zeros((model_w2v.vector_size,))
    words = text.split()
    words = [word for word in words if word in model_w2v]
    if words:
        return np.mean(model_w2v[words], axis=0)
    else:
        return np.zeros((model_w2v.vector_size,))


def build_and_train_model(X_train, y_train):
    model = Sequential([
        Dense(512, activation='relu', input_dim=100),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_dict = dict(enumerate(weights))

    model.fit(X_train, y_train, epochs=30, batch_size=32, class_weight=weights_dict, validation_split=0.2)
    return model


def generate_results(df, model, filepath):
    X_test = np.array([text_to_vector(text, w2v_model) for text in df['text']])
    predictions = (model.predict(X_test) > 0.5).astype(int)
    pd.DataFrame(predictions).to_csv(filepath, sep='\t', index=False, header=False)


X_train = np.array([text_to_vector(text, w2v_model) for text in train_df['text']])
y_train = train_df['label'].values

model = build_and_train_model(X_train, y_train)


generate_results(dev_df, model, "dev-0/out.tsv")
generate_results(test_df, model, "test-A/out.tsv")
