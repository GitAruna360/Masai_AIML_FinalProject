import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib

print("All modeling libraries imported successfully.")

df = pd.read_csv("titanic.csv")

print("Titanic CSV loaded successfully.")
print("Dataset shape:", df.shape)

display(df.head())

target = "survived"

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features].copy()
y = df[target].copy()

print("Features:")
print(features)

print("\nTarget:")
print(target)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("Class counts:")
display(y.value_counts())

print("\nClass percentages:")
display(
    (y.value_counts(normalize=True) * 100)
    .round(2)
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape :", X_test.shape)

print("\nTraining target distribution:")
display(
    (y_train.value_counts(normalize=True) * 100)
    .round(2)
)

print("\nTesting target distribution:")
display(
    (y_test.value_counts(normalize=True) * 100)
    .round(2)
)

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

print("Numeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)
print("Numeric transformer created successfully.")

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

print("Categorical transformer created successfully.")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

print("ColumnTransformer created successfully.")

X_train_transformed = preprocessor.fit_transform(X_train)

print(
    "Transformed training shape:",
    X_train_transformed.shape
)

feature_names = preprocessor.get_feature_names_out()

print("Number of transformed features:", len(feature_names))
print("\nFeature names:")
print(feature_names)

logistic_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

logistic_pipeline.fit(
    X_train,
    y_train
)

print("Logistic Regression trained successfully.")

decision_tree_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )
    ]
)

decision_tree_pipeline.fit(
    X_train,
    y_train
)

print("Decision Tree trained successfully.")

random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                oob_score=True
            )
        )
    ]
)

random_forest_pipeline.fit(
    X_train,
    y_train
)

print("Random Forest trained successfully.")

logistic_pred = logistic_pipeline.predict(X_test)
decision_tree_pred = decision_tree_pipeline.predict(X_test)
random_forest_pred = random_forest_pipeline.predict(X_test)

print("Predictions generated successfully.")
print("Logistic Regression predictions:", len(logistic_pred))
print("Decision Tree predictions:", len(decision_tree_pred))
print("Random Forest predictions:", len(random_forest_pred))

logistic_prob = logistic_pipeline.predict_proba(
    X_test
)[:, 1]

decision_tree_prob = decision_tree_pipeline.predict_proba(
    X_test
)[:, 1]

random_forest_prob = random_forest_pipeline.predict_proba(
    X_test
)[:, 1]

print("Probability predictions generated successfully.")

logistic_pipeline
decision_tree_pipeline
random_forest_pipeline

def evaluate_classifier(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(y_test, predictions)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "Confusion Matrix": cm
    }

models = {
    "Logistic Regression": logistic_pipeline,
    "Decision Tree": decision_tree_pipeline,
    "Random Forest": random_forest_pipeline
}

results = {}

for name, model in models.items():
    results[name] = evaluate_classifier(
        model,
        X_test,
        y_test
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy  : {results[name]['Accuracy']:.4f}")
    print(f"Precision : {results[name]['Precision']:.4f}")
    print(f"Recall    : {results[name]['Recall']:.4f}")
    print(f"F1 Score  : {results[name]['F1']:.4f}")
    print(f"AUC       : {results[name]['AUC']:.4f}")

    print("\nConfusion Matrix:")
    print(results[name]["Confusion Matrix"])

classification_results = pd.DataFrame({
    name: {
        "Accuracy": result["Accuracy"],
        "Precision": result["Precision"],
        "Recall": result["Recall"],
        "F1": result["F1"],
        "AUC": result["AUC"]
    }
    for name, result in results.items()
}).T

display(
    classification_results.round(4)
)

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4)
)

for ax, (name, model) in zip(
    axes,
    models.items()
):
    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "charts/classification_confusion_matrices.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 6))

for name, model in models.items():

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Classification Models")
plt.legend()

plt.tight_layout()

plt.savefig(
    "charts/classification_roc_curves.png",
    dpi=300
)

plt.show()

# Get the fitted preprocessing component from the Decision Tree pipeline
tree_preprocessor = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
)

# Get transformed feature names
tree_feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)

# Get the fitted Decision Tree model
tree_model = (
    decision_tree_pipeline
    .named_steps["classifier"]
)

# Plot the decision tree
plt.figure(figsize=(25, 15))

plot_tree(
    tree_model,
    feature_names=tree_feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=4,
    fontsize=8
)

plt.title("Decision Tree Classifier")

plt.tight_layout()

plt.savefig(
    "charts/decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Training class distribution:")
display(y_train.value_counts())

print("\nTraining class percentages:")
display(
    (y_train.value_counts(normalize=True) * 100).round(2)
)

print("\nTest class distribution:")
display(y_test.value_counts())

print("\nTest class percentages:")
display(
    (y_test.value_counts(normalize=True) * 100).round(2)
)

baseline_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

baseline_pipeline.fit(
    X_train,
    y_train
)

print("Baseline Random Forest trained.")

baseline_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

baseline_pipeline.fit(
    X_train,
    y_train
)

print("Baseline Random Forest trained.")

smote_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(random_state=42)
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

smote_pipeline.fit(
    X_train,
    y_train
)

print("SMOTE Random Forest trained.")

baseline_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

baseline_pipeline.fit(
    X_train,
    y_train
)

print("Baseline Random Forest trained successfully.")

baseline_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

baseline_pipeline.fit(
    X_train,
    y_train
)

print("Baseline Random Forest trained successfully.")

balanced_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

balanced_pipeline.fit(
    X_train,
    y_train
)

print("Class-weight balanced Random Forest trained successfully.")

smote_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(random_state=42)
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

smote_pipeline.fit(
    X_train,
    y_train
)

print("SMOTE Random Forest trained successfully.")

imbalance_models = {
    "Baseline": baseline_pipeline,
    "Class Weight Balanced": balanced_pipeline,
    "SMOTE": smote_pipeline
}

imbalance_results = []

for name, model in imbalance_models.items():

    predictions = model.predict(X_test)

    imbalance_results.append({
        "Strategy": name,
        "Precision": precision_score(
            y_test,
            predictions
        ),
        "Recall": recall_score(
            y_test,
            predictions
        ),
        "F1": f1_score(
            y_test,
            predictions
        )
    })

imbalance_results_df = pd.DataFrame(
    imbalance_results
)

display(
    imbalance_results_df.round(4)
)

print("baseline_pipeline" in globals())
print("balanced_pipeline" in globals())
print("smote_pipeline" in globals())

rf_pipeline_for_grid = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                oob_score=True
            )
        )
    ]
)

print("Random Forest GridSearch pipeline created.")

param_grid = {
    "classifier__n_estimators": [
        100,
        200,
        300
    ],
    "classifier__max_depth": [
        None,
        5,
        10,
        15
    ],
    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}

print("Parameter grid created.")

grid_search = GridSearchCV(
    estimator=rf_pipeline_for_grid,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    return_train_score=True
)

print("Starting GridSearchCV...")

grid_search.fit(
    X_train,
    y_train
)

print("GridSearchCV completed.")

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1 score:")
print(f"{grid_search.best_score_:.4f}")

best_rf_pipeline = grid_search.best_estimator_

best_rf_model = (
    best_rf_pipeline
    .named_steps["classifier"]
)

print("\nBest Random Forest OOB score:")
print(f"{best_rf_model.oob_score_:.4f}")

best_rf_predictions = best_rf_pipeline.predict(
    X_test
)

best_rf_probabilities = best_rf_pipeline.predict_proba(
    X_test
)[:, 1]

print("Tuned Random Forest test-set results:")
print(
    f"Accuracy  : {accuracy_score(y_test, best_rf_predictions):.4f}"
)
print(
    f"Precision : {precision_score(y_test, best_rf_predictions):.4f}"
)
print(
    f"Recall    : {recall_score(y_test, best_rf_predictions):.4f}"
)
print(
    f"F1 Score  : {f1_score(y_test, best_rf_predictions):.4f}"
)
print(
    f"AUC       : {roc_auc_score(y_test, best_rf_probabilities):.4f}"
)

regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X_reg = df[regression_features].copy()
y_reg = df["fare"].copy()

print("Regression features:")
print(regression_features)

print("\nRegression target:")
print("fare")

print("\nX_reg shape:", X_reg.shape)
print("y_reg shape:", y_reg.shape)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

print("Regression training shape:", X_reg_train.shape)
print("Regression testing shape :", X_reg_test.shape)

reg_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked"
]

reg_numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

reg_categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            reg_numeric_transformer,
            reg_numeric_features
        ),
        (
            "cat",
            reg_categorical_transformer,
            reg_categorical_features
        )
    ]
)

print("Regression preprocessing pipeline created.")

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)

print("Linear regression pipeline created.")

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

print("Linear regression trained successfully.")

y_reg_pred = regression_pipeline.predict(
    X_reg_test
)

print("Regression predictions generated.")
print("Number of predictions:", len(y_reg_pred))

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

reg_feature_names = (
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

p = len(reg_feature_names)
n = len(y_reg_test)

print("Number of transformed predictors (p):", p)
print("Number of test observations (n):", n)

adjusted_r2 = (
    1
    - ((1 - r2) * (n - 1))
    / (n - p - 1)
)

print(f"Adjusted R²: {adjusted_r2:.4f}")

regression_metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R²",
        "Adjusted R²"
    ],
    "Value": [
        mae,
        rmse,
        r2,
        adjusted_r2
    ]
})

display(
    regression_metrics.round(4)
)

residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=y_reg_pred,
    y=residuals
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Residual Plot — Linear Regression")

plt.tight_layout()

plt.savefig(
    "charts/regression_residuals.png",
    dpi=300
)

plt.show()

print("Residual summary:")
display(
    pd.Series(residuals).describe()
)

residual_diagnostic = pd.DataFrame({
    "Predicted Fare": y_reg_pred,
    "Residual": residuals,
    "Absolute Residual": np.abs(residuals)
})

print(
    "Correlation between predicted fare and absolute residuals:"
)

print(
    residual_diagnostic[
        ["Predicted Fare", "Absolute Residual"]
    ].corr().iloc[0, 1]
)

# Classification comparison table
classification_final = classification_results.copy()

display(
    classification_final.round(4)
)

regression_final = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R²",
        "Adjusted R²"
    ],
    "Linear Regression": [
        mae,
        rmse,
        r2,
        adjusted_r2
    ]
})

display(
    regression_final.round(4)
)

best_classifier_name = (
    classification_results["F1"].idxmax()
)

best_classifier_metrics = (
    classification_results.loc[
        best_classifier_name
    ]
)

print("Best classifier:", best_classifier_name)
print(best_classifier_metrics.round(4))

tuned_rf_metrics = {
    "Accuracy": accuracy_score(
        y_test,
        best_rf_predictions
    ),
    "Precision": precision_score(
        y_test,
        best_rf_predictions
    ),
    "Recall": recall_score(
        y_test,
        best_rf_predictions
    ),
    "F1": f1_score(
        y_test,
        best_rf_predictions
    ),
    "AUC": roc_auc_score(
        y_test,
        best_rf_probabilities
    )
}

display(
    pd.DataFrame(tuned_rf_metrics, index=["Tuned Random Forest"])
    .round(4)
)

best_rf_pipeline

print(best_rf_pipeline)

import joblib

full_pipeline = best_rf_pipeline

joblib.dump(
    full_pipeline,
    "best_pipeline.joblib"
)

print(
    "Complete fitted pipeline saved as best_pipeline.joblib"
)

loaded_pipeline = joblib.load(
    "best_pipeline.joblib"
)

print("Pipeline reloaded successfully.")
print(loaded_pipeline)

raw_sample = X_test.iloc[[0]].copy()

print("Raw input:")
display(raw_sample)

raw_prediction = loaded_pipeline.predict(
    raw_sample
)

print(
    "Prediction from reloaded pipeline:",
    raw_prediction
)

print(
    "Actual target:",
    y_test.iloc[0]
)

best_classifier_name = classification_results["F1"].idxmax()

print("Best classifier:", best_classifier_name)

display(
    classification_results.round(4)
)

tuned_rf_metrics = {
    "Accuracy": accuracy_score(
        y_test,
        best_rf_predictions
    ),
    "Precision": precision_score(
        y_test,
        best_rf_predictions
    ),
    "Recall": recall_score(
        y_test,
        best_rf_predictions
    ),
    "F1": f1_score(
        y_test,
        best_rf_predictions
    ),
    "AUC": roc_auc_score(
        y_test,
        best_rf_probabilities
    )
}

display(
    pd.DataFrame(
        tuned_rf_metrics,
        index=["Tuned Random Forest"]
    ).round(4)
)

display(
    regression_final.round(4)
)

full_pipeline = best_rf_pipeline

joblib.dump(
    full_pipeline,
    "best_pipeline.joblib"
)

print("Complete fitted pipeline saved successfully.")

loaded_pipeline = joblib.load(
    "best_pipeline.joblib"
)

print("Pipeline reloaded successfully.")

raw_sample = X_test.iloc[[0]].copy()

print("Raw input:")
display(raw_sample)

raw_prediction = loaded_pipeline.predict(
    raw_sample
)

print("Prediction:", raw_prediction)
print("Actual:", y_test.iloc[0])
