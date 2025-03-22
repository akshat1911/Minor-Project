import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import wfdb  # For loading and working with PhysioNet data
from sklearn.preprocessing import LabelEncoder  # For encoding labels
from sklearn.model_selection import train_test_split  # For splitting the data

# TensorFlow and Keras for building and training the neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input

# Load the record from PhysioNet
record_name = '100'
database_name = 'mitdb'
record = wfdb.rdrecord(record_name, pn_dir=database_name)
annotation = wfdb.rdann(record_name, 'atr', pn_dir=database_name)

print("Signal shape:", record.p_signal.shape)
print("Annotation symbols:", annotation.symbol[:10])

def get_segments(signal, annotations, window_size=360):
    segments = []
    labels = []
    for idx, symbol in enumerate(annotations.symbol):
        center = annotations.sample[idx]
        if center < window_size or center > signal.shape[0] - window_size:
            continue  # Skip beats too close to the start or end
        segment = signal[center - window_size:center + window_size]
        segments.append(segment)
        labels.append(symbol)
    return np.array(segments), np.array(labels)

# Get segments and labels
segments, labels = get_segments(record.p_signal[:, 0], annotation)  # Using only channel 1 for simplicity

print("Number of segments:", len(segments))
print("Example segment shape:", segments[0].shape)

# Initialize the encoder
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(segments, encoded_labels, test_size=0.2, random_state=42)

# Define the neural network structure
model = Sequential([
    Input(shape=(720,)),  # Explicitly define the input shape here
    Flatten(),            # Now Flatten doesn't need the input_shape
    Dense(128, activation='relu'),  # First dense layer
    Dense(64, activation='relu'),   # Second dense layer
    Dense(len(set(encoded_labels)), activation='softmax')  # Output layer with one neuron per class
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.6f}")

# Plotting training and validation accuracy/loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred_classes))

# ROC Curve for multi-class classification
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
y_pred_bin = model.predict(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for class {le.classes_[i]}')
    plt.legend(loc="lower right")
    plt.show()

# Function to plot ECG signal with annotations
def plot_ecg_with_annotation(index):
    plt.figure(figsize=(14, 5))
    plt.plot(X_test[index], label=f"Predicted: {le.inverse_transform([y_pred_classes[index]])[0]}, True: {le.inverse_transform([y_test[index]])[0]}")
    plt.title('ECG Signal with Model Prediction')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Plot for a specific index
plot_ecg_with_annotation(10)  # Change index as needed
