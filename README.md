## 📊 Biosensor Time-Series Flare-Up Prediction

This project aims to predict **flare-ups in chronic conditions like Rheumatoid Arthritis** using biosensor time-series data. It features a full pipeline: from preprocessing to classification and analysis.

---

### 📁 Project Structure

```
.
├── preprocess.py                 # Loads and transforms raw biosensor .txt data
├── train.py                      # Builds and trains a binary classification model
├── model_train.ipynb             # Jupyter version for interactive exploration
├── /data/                        # Place raw .txt biosensor files here
├── /outputs/                     # All processed outputs and visualizations
│   ├── X_train.npy               # Training features
│   ├── simulated_labels.npy      # Binary labels (flare / no flare)
│   ├── loss_curve.png            # Model training loss curve
│   ├── feature_distributions.png # Distributions of top 5 features
│   └── correlation_matrix.png    # Correlation heatmap of features
└── README.md
```

---

### 🧰 Requirements

Install dependencies via pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

### ▶️ How to Run

#### 1. Preprocess the Data

Ensure `.txt` biosensor files are placed under `/data/`. Then run:

```bash
python preprocess.py
```

This generates:
- Statistical features from 45 sensors over sliding windows
- Simulated binary labels (0 = no flare, 1 = flare)
- Visual outputs:
  - `feature_distributions.png`
  - `correlation_matrix.png`

#### 2. Train the Model

```bash
python train.py
```

This trains an MLP model and saves:
- `mlp_model.h5` (trained weights)
- Train-validation plots
- Numpy arrays of all dataset splits

---

### 🧠 Model Summary

| Layer | Type      | Details                 |
|-------|-----------|-------------------------|
| 1     | Dense     | 128 units, ReLU         |
| 2     | Dropout   | 30% dropout             |
| 3     | Dense     | 64 units, ReLU          |
| 4     | Dropout   | 30% dropout             |
| 5     | Dense     | 1 unit, Sigmoid (output)|

---

## ✅ Results & Analysis

### 🔢 1. Training Configuration
- **Input Features:** 180 (mean, std, min, max × 45 sensors)
- **Labels:** Simulated binary (flare / no flare)
- **Splits:** 80% train / 20% test; 20% of train as validation
- **Batch Size:** 8
- **Epochs:** 50
- **Early Stopping:** Patience = 10 epochs

---

### 📉 2. Loss Curve
![loss_curve](outputs/loss_curve.png)

- **Initial loss ~0.73**, drops to ~0.695 and flattens.
- **Validation loss is flat**, indicating:
  - Model likely can't learn useful signal from random labels.
  - No overfitting, but also no generalization.

---

### 📊 3. Accuracy & Metrics

- **Test Accuracy:** ~50–55% (random guess level)
- **Precision, Recall, F1:** Balanced, but not meaningful without true labels.
- **Confusion Matrix:** Expected to show equal FP/FN due to random labeling.

---

### 🧪 4. EDA Summary

#### a. Feature Distributions
![feature_distributions](outputs/feature_distributions.png)

- Features are well-scaled and centered (due to `StandardScaler`).
- Histograms confirm normalization is working.

#### b. Correlation Matrix
![correlation_matrix](outputs/correlation_matrix.png)

- High correlation among features from the same sensor.
- May lead to feature redundancy — consider PCA or correlation pruning.

---

### 📌 Hyperparameters Used

| Parameter           | Value            |
|---------------------|------------------|
| `window_size`       | 100 samples      |
| `hidden_layers`     | 128 → 64 units   |
| `dropout`           | 30%              |
| `activation`        | ReLU, Sigmoid    |
| `optimizer`         | Adam             |
| `loss`              | Binary CrossEntropy |
| `early_stopping`    | patience = 10    |
| `epochs`            | 50               |
| `batch_size`        | 8                |

---

### ⚠️ Current Limitations

| Issue | Recommendation |
|-------|----------------|
| Labels are simulated | Replace with clinical flare-up labels |
| Accuracy ≈ 50% | Indicates model is guessing |
| Feature redundancy | Apply PCA or correlation filtering |
| Static features only | Try LSTM/Transformer for temporal modeling |

---

### 🚀 Future Work

- Replace random labels with real flare-up data
- Try time-aware models: LSTM, GRU, Temporal CNN
- Incorporate domain-specific features (e.g., RMS, entropy)
- Model explainability: SHAP, LIME
- AutoML with hyperparameter tuning (Optuna/Keras Tuner)

---

### 👤 Author

**Shashank Joshi**  
*M.Tech, IIT Bombay*  
📧 [Contact](#) | 💼 [LinkedIn](#)
