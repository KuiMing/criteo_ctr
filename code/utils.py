import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Flatten,
    Dropout,
    Activation,
    Add,
    Concatenate,
    Multiply,
    Lambda,
    Subtract,
)
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import gc
import pickle


class DataProcessor:
    def __init__(self, dense_feeats: list, sparse_feeats: list):
        self.sparse_encoder = dict()
        self.dense_feeats = dense_feeats
        self.sparse_feats = sparse_feeats

    def process_dense_feats(self, data: pd.DataFrame) -> pd.DataFrame:
        feats = self.dense_feeats
        df = data.copy()
        for f in feats:
            df[f] = df[f].fillna(df[f].mode()[0])
            df.loc[df[f] > -1, f] = np.log(df.loc[df[f] > -1, f] + 1)
            df.loc[df[f] <= -1, f] = -1
            gc.collect()
        return df[feats]

    def process_sparse_feats(self, data: pd.DataFrame) -> pd.DataFrame:
        feats = self.sparse_feats
        df = data.copy()
        df[feats] = df[feats].fillna("-1")
        for f in feats:
            label_encoder = LabelEncoder()
            labels = df[f].unique().tolist()
            labels.append("-1")  # Add a special category for unknown labels
            label_encoder.fit(list(set(labels)))
            df[f] = label_encoder.transform(df[f])
            self.sparse_encoder[f] = label_encoder
            gc.collect()
        return df[feats]

    def transform_sparse_feats(self, data: pd.DataFrame) -> pd.DataFrame:
        feats = self.sparse_feats
        df = data.copy()
        df[feats] = df[feats].fillna("-1")
        for f in feats:
            known_labels = set(self.sparse_encoder[f].classes_)
            df.loc[~df[f].isin(known_labels), f] = "-1"
            df[f] = self.sparse_encoder[f].transform(df[f])
        return df[feats]


class DeepFM(DataProcessor):
    def design_model(self, data: pd.DataFrame) -> Model:
        # Inputs of Dense Features
        dense_inputs = []
        for f in self.dense_feats:
            _input = Input([1], name=f)
            dense_inputs.append(_input)
        # Concatenate Input of Dense Features
        concat_dense_inputs = Concatenate(axis=1)(dense_inputs)  # ?, 13
        # Connect to a dense layer
        fst_order_dense_layer = Dense(1)(concat_dense_inputs)  # ?, 1

        # Inputs of Sparse Features
        sparse_inputs = []
        for f in self.sparse_feats:
            _input = Input([1], name=f)
            sparse_inputs.append(_input)

        sparse_1d_embed = []
        for i, _input in enumerate(sparse_inputs):
            f = self.sparse_feats[i]
            voc_size = data[f].nunique()
            # L2 regularizer
            reg = tf.keras.regularizers.l2(0.5)
            _embed = Embedding(voc_size, 1, embeddings_regularizer=reg)(_input)
            # Embedding --> Flatten
            _embed = Flatten()(_embed)
            sparse_1d_embed.append(_embed)
        # some of each embedding
        fst_order_sparse_layer = Add()(sparse_1d_embed)

        linear_part = Add()([fst_order_dense_layer, fst_order_sparse_layer])
        # embedding size: k
        k = 8

        # embedding of sparse inputs
        sparse_kd_embed = []
        for i, _input in enumerate(sparse_inputs):
            voc_size = data[self.sparse_feats[i]].nunique()
            reg = tf.keras.regularizers.l2(0.7)
            _embed = Embedding(voc_size, k, embeddings_regularizer=reg)(_input)
            sparse_kd_embed.append(_embed)

        # Concatenate embedding of sparse inputs--> (n, k), n: feature number
        concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)  # ?, n, k

        # sum -> square
        sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(
            concat_sparse_kd_embed
        )  # ?, k
        square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])  # ?, k

        # square -> sum
        square_kd_embed = Multiply()(
            [concat_sparse_kd_embed, concat_sparse_kd_embed]
        )  # ?, n, k
        sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(
            square_kd_embed
        )  # ?, k

        # (square_sum_kd_embed - sum_square_kd_embed ) / 2
        sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])  # ?, k
        sub = Lambda(lambda x: x * 0.5)(sub)  # ?, k
        snd_order_sparse_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(
            sub
        )  # ?, 1

        flatten_sparse_embed = Flatten()(concat_sparse_kd_embed)  # ?, n*k

        fc_layer = Dropout(0.5)(
            Dense(256, activation="relu")(flatten_sparse_embed)
        )  # ?, 256
        fc_layer = Dropout(0.3)(Dense(256, activation="relu")(fc_layer))  # ?, 256
        fc_layer = Dropout(0.1)(Dense(256, activation="relu")(fc_layer))  # ?, 256

        fc_layer_output = Dense(1)(fc_layer)  # ?, 1
        output_layer = Add()([linear_part, snd_order_sparse_layer, fc_layer_output])
        output_layer = Activation("sigmoid")(output_layer)

        model = Model(dense_inputs + sparse_inputs, output_layer)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_crossentropy", tf.keras.metrics.AUC(name="auc")],
        )
        return model


def data_generator(df, batch_size, feature):
    while True:
        for start in range(0, df.shape[0], batch_size):
            end = min(start + batch_size, df.shape[0])
            subset = df.iloc[start:end]
            inputs = [subset[feat].values for feat in feature]
            yield inputs, subset.label.values


def predict(data_path, encoder_path, model_path, result_path):
    test_df_cols = [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    data = pd.read_csv(data_path, sep="\t", names=test_df_cols)
    model = load_model(model_path)
    deepfm_processor = pickle.load(open(encoder_path))
    dense_feats = deepfm_processor.dense_feats
    sparse_feats = deepfm_processor.sparse_feats
    data[sparse_feats] = deepfm_processor.transform_sparse_feats(data)
    data[dense_feats] = deepfm_processor.process_dense_feats(data)

    test_dense_x = [data[f].values for f in dense_feats]
    test_sparse_x = [data[f].values for f in sparse_feats]
    y_test_pred = model.predict(test_dense_x + test_sparse_x, batch_size=2**18)

    submision = pd.DataFrame(
        dict(
            Id=list(range(len(y_test_pred))),
            Predicted=y_test_pred.reshape(len(y_test_pred)),
        )
    )
    submision["Id"] += 60000000
    submision.to_csv(result_path, index=False)
