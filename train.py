import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# -----------------------------
# STEP 1: Load Preprocessed Data
# -----------------------------
output_dir = r"C:\Users\pshas\OneDrive\Desktop\AcademicResearch\IITBombay\placement\DreptoAIML\Project-1\outputs"
X = np.load(os.path.join(output_dir, "preprocessed_features.npy"))
y = np.load(os.path.join(output_dir, "simulated_labels.npy"))

print("[INFO] Loaded preprocessed data.")
print(f"[INFO] Feature shape: {X.shape}, Label shape: {y.shape}")

# -----------------------------
# STEP 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("[INFO] Data split: ", X_train.shape, X_test.shape)

# Second split: Train vs Val
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# -----------------------------
# Save for reuse in model_test.py
# -----------------------------
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

# -----------------------------
# STEP 3: Build MLP Model
# -----------------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# STEP 4: Train Model
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# STEP 5: Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"[RESULT] Test Accuracy: {accuracy:.4f}")

# Classification report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# -----------------------------
# STEP 6: Save Model & Plots
# -----------------------------
model.save(os.path.join(output_dir, "mlp_model.h5"))

# Plot loss
plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()

# Plot accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
plt.close()

print(f"[INFO] Model and plots saved to {output_dir}/")
