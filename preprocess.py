import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# STEP 1: Time-Series Feature Extraction
# -------------------------------
class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=100):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure input is a DataFrame for .iloc to work
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_windows = X.shape[0] // self.window_size
        features = []
        for i in range(n_windows):
            window = X.iloc[i*self.window_size:(i+1)*self.window_size]
            stats = window.describe().loc[['mean', 'std', 'min', 'max']].values.flatten()
            features.append(stats)

        return np.array(features)

# -------------------------------
# STEP 2: Load Individual File
# -------------------------------
def load_txt_file(filepath):
    return pd.read_csv(filepath, header=None)

# -------------------------------
# STEP 3: Recursively Load All Files
# -------------------------------
def load_all_subjects(data_dir):
    all_subjects = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root, file) 
                df = load_txt_file(full_path)
                all_subjects.append(df)
    return all_subjects

# -------------------------------
# STEP 4: Preprocessing Pipeline
# -------------------------------
def create_pipeline(window_size=100):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_extractor', TimeSeriesFeatureExtractor(window_size=window_size))
    ])
    return pipeline

# -------------------------------
# STEP 5: Apply Pipeline to All Subjects
# -------------------------------
def preprocess_all_subjects(subject_dfs, pipeline):
    all_features = []
    for df in subject_dfs:
        features = pipeline.fit_transform(df)
        all_features.append(features)
    return np.vstack(all_features)

# -------------------------------
# OUTPUT
# -------------------------------
output_dir = r"C:\Users\pshas\OneDrive\Desktop\AcademicResearch\IITBombay\placement\DreptoAIML\Project-1\outputs"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    data_folder = r"C:\Users\pshas\OneDrive\Desktop\AcademicResearch\IITBombay\placement\DreptoAIML\Project-1\data"
    
    print("[INFO] Loading raw biosensor files...")
    subject_dfs = load_all_subjects(data_folder)
    print(f"[INFO] Loaded {len(subject_dfs)} subject sessions.")

    print("[INFO] Creating preprocessing pipeline...")
    pipeline = create_pipeline(window_size=100)

    print("[INFO] Running pipeline and extracting features...")
    X = preprocess_all_subjects(subject_dfs, pipeline)
    print(f"[INFO] Final preprocessed feature matrix shape: {X.shape}")

    # -------------------------------
    # STEP 6: Simulate Labels for Testing
    # -------------------------------
    np.random.seed(42)
    y = np.random.randint(0, 2, size=X.shape[0])  # Binary labels (flare/no flare)

    # -------------------------------
    # Define Feature Names
    # -------------------------------
    n_sensors = X.shape[1] // 4 
    feature_names = []
    stats = ['mean', 'std', 'min', 'max']
    for i in range(n_sensors):  # 45 sensors
        for stat in stats:
            feature_names.append(f"sensor_{i}_{stat}")

    # -------------------------------
    # Save Outputs
    # -------------------------------
    np.save(os.path.join(output_dir, "preprocessed_features.npy"), X)
    np.save(os.path.join(output_dir, "simulated_labels.npy"), y)

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    df.to_csv(os.path.join(output_dir, "preprocessed_features.csv"), index=False)
    print(f"[INFO] Saved .npy and .csv in {output_dir}/")

    # -------------------------------
    # STEP 8: Basic EDA
    # -------------------------------

    print("\n--- Exploratory Data Analysis ---")
    print("Shape:", df.shape)
    print("Summary Statistics:\n", df.describe().T)

    # Plot 5 sample features
    sample_cols = df.columns[:5]
    df[sample_cols].hist(figsize=(12, 6))
    plt.suptitle("Distribution of First 5 Features")
    plt.xlabel("Sensor Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"))
    plt.close()

    # Feature correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.iloc[:, :-1].corr(), cmap='coolwarm', center=0)
    plt.title("Feature Correlation Matrix")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()

    print(f"[INFO] EDA plots saved to {output_dir}/")




