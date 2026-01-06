import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# --- CONSTANTS ---
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CLASSES = 46
EPOCHS = 10
BATCH_SIZE = 80
# Using relative paths is better for portability
TRAIN_PATH = os.path.join('.', 'data', 'train')
TEST_PATH = os.path.join('.', 'data', 'test')

def load_data(path):
    """
    Loads images and labels from a directory where subfolders represent classes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}. Please check your folder structure.")

    data = []
    labels = []
    label_map = {} # To store {0: 'character_name'} mappings
    
    # Sort folders to ensure consistent labeling across different runs/machines
    folders = sorted(os.listdir(path))
    
    for label_idx, folder_name in enumerate(folders):
        label_map[label_idx] = folder_name
        folder_path = os.path.join(path, folder_name)
        
        # Iterate through images
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            # Read as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize to ensure consistency (defensive coding)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                data.append(img)
                labels.append(label_idx)
        
        print(f"Loaded class {label_idx + 1}/{len(folders)}: {folder_name}")

    return np.array(data), np.array(labels), label_map

def preprocess_data(data, labels, mode='cnn'):
    """
    Normalizes pixel values and reshapes data based on model type.
    """
    # Normalize to [0, 1]
    data = data.astype('float32') / 255.0
    
    if mode == 'mlp':
        # Flatten for Dense layers: (N, 1024)
        data = data.reshape((-1, IMG_WIDTH * IMG_HEIGHT))
    elif mode == 'cnn':
        # Reshape for Conv2D: (N, 32, 32, 1)
        data = data.reshape((-1, IMG_WIDTH, IMG_HEIGHT, 1))
        
    return data, labels

def build_mlp_model():
    """Builds a simple Multi-Layer Perceptron."""
    model = models.Sequential([
        layers.Input(shape=(IMG_WIDTH * IMG_HEIGHT,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        # Softmax is standard for multi-class classification
        layers.Dense(NUM_CLASSES, activation='softmax') 
    ], name="MLP_Model")
    return model

def build_cnn_model():
    """Builds a Convolutional Neural Network."""
    model = models.Sequential([
        layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        
        # Block 1
        layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
        layers.MaxPool2D(pool_size=(2,2)),
        
        # Block 2
        layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Classification Head
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name="CNN_Model")
    return model

def plot_training_history(history, model_name):
    """Visualizes accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} - Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} - Loss')
    
    plt.show()

def main():
    # 1. Load Data
    print("Loading Training Data...")
    train_data, train_labels, label_map = load_data(TRAIN_PATH)
    print("Loading Test Data...")
    test_data, test_labels, _ = load_data(TEST_PATH)

    # 2. Shuffle Training Data (Good practice)
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # 3. Choose Model Type: 'cnn' or 'mlp'
    MODEL_TYPE = 'cnn' 
    print(f"Preparing data for {MODEL_TYPE.upper()} model...")
    
    x_train, y_train = preprocess_data(train_data, train_labels, mode=MODEL_TYPE)
    x_test, y_test = preprocess_data(test_data, test_labels, mode=MODEL_TYPE)

    # 4. Build Model
    if MODEL_TYPE == 'cnn':
        model = build_cnn_model()
    else:
        model = build_mlp_model()
    
    model.summary()

    # 5. Compile
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001), # Standard LR
        metrics=["accuracy"]
    )

    # 6. Train
    print("Starting Training...")
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # 7. Evaluate
    print("\nEvaluating on Test Set...")
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Final Test Accuracy: {acc*100:.2f}%")

    # 8. Visualize
    plot_training_history(history, model.name)

    # 9. Save Model
    model.save(f'devnagri_{MODEL_TYPE}.h5')
    print(f"Model saved as devnagri_{MODEL_TYPE}.h5")

if __name__ == "__main__":
    main()