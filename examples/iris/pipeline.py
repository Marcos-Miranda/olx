from pathlib import Path

from pydantic import BaseModel

from mlbox import Pipeline, get_context, step


class InputParams(BaseModel):
    """Parameters of the pipeline, given with `-p key=value`."""

    dataset_path: str = "iris.csv"
    test_size: float = 0.7
    n_estimators: int = 2


# The libraries from requirements.txt are only installed in the step containers,
# so they are imported inside the steps rather than at module level.


@step()
def preprocess() -> dict[str, Path]:
    """Split the dataset into train and test CSVs and return their paths."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    ctx = get_context(InputParams)
    df = pd.read_csv(ctx.input_params.dataset_path)
    train, test = train_test_split(
        df, test_size=ctx.input_params.test_size, random_state=ctx.random_seed, stratify=df["target"]
    )
    train_path, test_path = ctx.data_path / "train.csv", ctx.data_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return {"train": train_path, "test": test_path}


@step(cpus=2, memory="2g")
def train_random_forest(train_path: str) -> Path:
    """Train a random forest and return the path of the saved model."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    ctx = get_context(InputParams)
    train = pd.read_csv(train_path)
    model = RandomForestClassifier(n_estimators=ctx.input_params.n_estimators, random_state=ctx.random_seed)
    model.fit(train.drop(columns="target"), train["target"])
    model_path = ctx.data_path / "random_forest.joblib"
    joblib.dump(model, model_path)
    return model_path


@step()
def train_logistic_regression(train_path: str) -> Path:
    """Train a logistic regression, with missing values imputed first, and return the path of the saved model."""
    import joblib
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    ctx = get_context(InputParams)
    train = pd.read_csv(train_path)
    model = make_pipeline(
        SimpleImputer(strategy="median"), LogisticRegression(max_iter=500, random_state=ctx.random_seed)
    )
    model.fit(train.drop(columns="target"), train["target"])
    model_path = ctx.data_path / "logistic_regression.joblib"
    joblib.dump(model, model_path)
    return model_path


@step()
def evaluate(test_path: str, random_forest_path: str, logistic_regression_path: str) -> dict[str, float]:
    """Log each model's test accuracy as an MLflow metric and return them."""
    import joblib
    import pandas as pd

    ctx = get_context(InputParams)
    test = pd.read_csv(test_path)
    features, target = test.drop(columns="target"), test["target"]
    models = {
        "random_forest": joblib.load(random_forest_path),
        "logistic_regression": joblib.load(logistic_regression_path),
    }
    accuracies = {}
    for name, model in models.items():
        accuracies[name] = float(model.score(features, target))
        ctx.mlflow.log_metric(f"{name}_accuracy", accuracies[name])
    best_model = max(accuracies, key=accuracies.get)
    ctx.register_model(models[best_model], input_data=features, predict_function="predict_proba")
    return accuracies


data = preprocess()
random_forest = train_random_forest(data["train"])
logistic_regression = train_logistic_regression(data["train"])
report = evaluate(data["test"], random_forest, logistic_regression)

pipeline = Pipeline("iris", InputParams, steps=[evaluate], requirements="requirements.txt")
