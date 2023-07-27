import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import time


def gbm_grid_search(X, y, param_grid):
    # set up initial parameters
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
    }

    model = lgb.LGBMClassifier()
    # grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_log_loss")
    grid_search.fit(X, y)

    # Check Result
    results = grid_search.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(params, "had the following log loss:", -mean_score)

    return grid_search.best_params_


def gbm_train(
    best_params: dict, path: str, use_cols: list, read_rows: int
) -> lgb.basic.Booster:
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "force_col_wise": True,
    }
    params = {**params, **best_params}

    chunksize = 10**4  # adjust based on your system's memory
    num_boost_round = 1000
    importance_split = []
    importance_gain = []

    for i, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            names=use_cols,
            chunksize=chunksize,
            nrows=read_rows,
        )
    ):
        # Separate the features and the target variable
        X = chunk.drop("label", axis=1)
        y = chunk["label"]

        # Convert categorical features to `category` type
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category")

        # Split the data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Build dataset
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        # Train the model
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_val],
            early_stopping_rounds=100,
            init_model=gbm,
        )
        importance_split.append(
            gbm.feature_importance(importance_type="split", iteration=None)
        )
        importance_gain.append(
            gbm.feature_importance(importance_type="gain", iteration=None)
        )
    use_cols.remove("label")
    imp_split = pd.DataFrame(importance_split, columns=use_cols)
    imp_gain = pd.DataFrame(importance_gain, columns=use_cols)
    imp_gain.to_csv("importance_gain.csv", index=False)
    imp_split.to_csv("importance_split.csv", index=False)
    print("Save Importancs")

    return gbm


def predict_test(gbm: lgb.basic.Booster, path: str, test_df_cols: list) -> None:
    test_df = pd.read_csv(path, sep="\t", names=test_df_cols)
    for col in test_df.select_dtypes(include=["object"]).columns:
        test_df[col] = test_df[col].astype("category")
    y_test_pred = gbm.predict(test_df)

    submision = pd.DataFrame(
        dict(
            Id=list(range(len(y_test_pred))),
            Predicted=y_test_pred.reshape(len(y_test_pred)),
        )
    )
    submision["Id"] += 60000000
    submision.to_csv("submision.csv", index=False)


def main():
    dense_feats = [f"I{i}" for i in range(1, 14)]
    sparse_feats = [f"C{i}" for i in range(1, 27)]
    train_df_cols = ["label"] + dense_feats + sparse_feats
    test_df_cols = dense_feats + sparse_feats

    data = pd.read_csv("train.txt", sep="\t", names=train_df_cols, nrows=10**4)
    X = data.drop("label", axis=1)
    y = data["label"]
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    # set up parameter grids
    param_grid = {
        "feature_fraction": [0.6, 0.7, 0.8, 0.9],
        "num_leaves": list(range(10, 101)),
        "learning_rate": [0.05, 0.01, 0.005],
    }

    best_params = gbm_grid_search(X, y, param_grid)

    temp = pd.read_csv("train.txt", sep="\t", names=train_df_cols, usecols=["label"])
    read_rows = int(len(temp) * 0.5)

    gbm = gbm_train(best_params, "train.csv", train_df_cols, read_rows)

    predict_test(gbm, "test.txt", test_df_cols)


if __name__ == "__main__":
    main()
