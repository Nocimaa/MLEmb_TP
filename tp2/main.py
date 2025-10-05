import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("insurance.csv")
print(df.head())

X = df.drop("charges", axis=1)
y = df["charges"]

categorical_features = ["sex", "smoker", "region"]
numeric_features = ["age", "bmi", "children"]

categorical_transformer = Pipeline(
    steps=[("encoder", OneHotEncoder(drop="first", sparse_output=False))]
)
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
pipeline_lr = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))]
)




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline_lr.fit(X_train, y_train)

accuracy = pipeline_lr.score(X_test, y_test)

print(f"Model accuracy: {accuracy}")
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("MlEmb-Insurance")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    # mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Infer the model signature
    signature = infer_signature(X_train, pipeline_lr.predict(X_train))

    mlflow.log_artifact("encoders.pkl", artifact_path="label_encoders")

    # Log the model, which inherits the parameters and metric
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline_lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="InsuranceModel"
    )

    # Set a tag that we can use to remind ourselves what this model was for
    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "Basic LR model for insurance data"}
    )