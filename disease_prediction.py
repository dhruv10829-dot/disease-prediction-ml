import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.impute import SimpleImputer

sns.set(style='whitegrid')
plt.style.use('seaborn-v0_8-darkgrid')


def load_dataset(filepath):
    return pd.read_csv(filepath)


def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df


def encode_categorical_data(df):
    le = LabelEncoder()
    if 'Status' in df.columns:
        df['Status'] = le.fit_transform(df['Status'])
    if 'Country' in df.columns:
        df['Country'] = le.fit_transform(df['Country'])
    return df


def plot_correlation_heatmap(df, title="Feature Correlation Heatmap"):
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Correlation heatmap saved as 'correlation_heatmap.png'")
    plt.close()


def plot_roc_curve(y_test, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"ROC curve for {model_name} saved as 'roc_curve_{model_name.lower().replace(' ', '_')}.png'")
    plt.close()


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred


def main():
    print("="*60)
    print("DISEASE PREDICTION - MACHINE LEARNING PROJECT")
    print("="*60)

    print("\n[Step 1] Loading dataset...")
    df = load_dataset('disease_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\n[Step 2] Handling missing values...")
    df = handle_missing_values(df)
    print("Missing values handled.")

    print("\n[Step 3] Encoding categorical data...")
    df = encode_categorical_data(df)
    print("Categorical data encoded.")

    if 'Disease Risk' not in df.columns:
        print("Warning: 'Disease Risk' column not found. Using last column as target.")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df.drop('Disease Risk', axis=1)
        y = df['Disease Risk']

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    print("\n[Step 4] Splitting data into train and test sets (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    print("\n[Step 5] Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling applied.")

    print("\n[Plot] Generating correlation heatmap...")
    plot_correlation_heatmap(df)

    print("\n[Step 6] Training models...")

    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    print("Logistic Regression trained.")

    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    print("Random Forest trained.")

    print("\n[Step 7] Evaluating models...")

    evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")

    evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")

    print("\n[Step 8] Generating ROC curves...")

    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    plot_roc_curve(y_test, y_prob_lr, "Logistic Regression")
    plot_roc_curve(y_test, y_prob_rf, "Random Forest")

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    acc_lr = accuracy_score(y_test, lr_model.predict(X_test_scaled))
    acc_rf = accuracy_score(y_test, rf_model.predict(X_test_scaled))

    print(f"\nLogistic Regression Accuracy: {acc_lr:.4f}")
    print(f"Random Forest Accuracy:      {acc_rf:.4f}")

    if acc_rf > acc_lr:
        print(f"\nRandom Forest performs better by {(acc_rf - acc_lr):.4f} points")
    elif acc_lr > acc_rf:
        print(f"\nLogistic Regression performs better by {(acc_lr - acc_rf):.4f} points")
    else:
        print("\nBoth models have the same accuracy")

    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  - correlation_heatmap.png")
    print("  - roc_curve_logistic_regression.png")
    print("  - roc_curve_random_forest.png")


if __name__ == "__main__":
    main()
