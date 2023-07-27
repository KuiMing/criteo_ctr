from utils import DataProcessor, DeepFM, data_generator
import pandas as pd
import gc
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, log_loss
import pickle


import click


@click.command()
@click.option("-e", "encoder_path", help="encoding path")
@click.option("-m", "model_path", help="model path")
@click.option("-d", "data_path", help="data path")
def main(data_path, encoder_path, model_path):
    dense_feats = [f"I{i}" for i in range(1, 14)]
    sparse_feats = [f"C{i}" for i in range(1, 27)]
    train_df_cols = ["label"] + dense_feats + sparse_feats
    data = pd.read_csv(data_path, sep="\t", names=train_df_cols, nrows=10000)
    gc.collect()

    ## Remove not important features
    importance_split = pd.read_csv("importance_split.csv")
    importance_gain = pd.read_csv("importance_gain.csv")
    missing_count = data.isnull().mean()
    feat_status = pd.concat(
        [(importance_split == 0).sum(), (importance_gain == 0).sum(), missing_count],
        axis=1,
    )
    feat_status.columns = [
        "importance_split_zero_count",
        "importance_gain_zero_count",
        "null_percentage",
    ]
    removed_col = feat_status[
        (feat_status.importance_gain_zero_count > 100)
        & (feat_status.importance_split_zero_count > 100)
        & (feat_status.null_percentage > 0.4)
    ].index

    for i in removed_col.to_list():
        try:
            dense_feats.remove(i)
        except:
            sparse_feats.remove(i)

    ## Data Preprocessing
    deepfm = DeepFM(dense_feats, sparse_feats)
    data[dense_feats] = deepfm.process_dense_feats(data)
    data[sparse_feats] = deepfm.process_sparse_feats(data)
    gc.collect()
    print("Data Preprocessing Done")

    with open(encoder_path, "wb") as f:
        pickle.dump(deepfm, f)
    f.close()

    ## Model Building
    model = deepfm.design_model(data)

    batch_size = 2**18

    ## Data Generator
    train_generator = data_generator(
        data[: int(len(data) * 0.8)], batch_size, dense_feats + sparse_feats
    )
    val_generator = data_generator(
        data[int(len(data) * 0.8) : int(len(data) * 0.98)],
        batch_size,
        dense_feats + sparse_feats,
    )

    ## Callback
    stop_rule = EarlyStopping(monitor="val_loss", patience=50)
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, mode="min", verbose=1, min_lr=1e-7
    )

    ## Train
    history = model.fit(
        train_generator,
        steps_per_epoch=len(data[: int(len(data) * 0.8)]) // batch_size,
        epochs=1000,
        validation_data=val_generator,
        validation_steps=len(data[int(len(data) * 0.8) :]) // batch_size,
        callbacks=[stop_rule, reduce_lr],
    )
    print("Save Model")
    model.save(model_path)

    ## Evaluate
    print("==========Evaluate Testing Data==========")
    test_dense_x = [data[f].values[int(len(data) * 0.98) :] for f in dense_feats]
    test_sparse_x = [data[f].values[int(len(data) * 0.98) :] for f in sparse_feats]
    y = data.loc[int(len(data) * 0.98) :, "label"].values
    y_test_pred = model.predict(test_dense_x + test_sparse_x, batch_size)
    y_test_pred.reshape(len(y_test_pred))

    logloss = log_loss(y, y_test_pred)
    accuracy = accuracy_score(y, y_test_pred > 0.5)

    print(f"Log Loss: {logloss}")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
