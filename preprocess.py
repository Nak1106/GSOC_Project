import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Path to your dataset
base_dir = r"C:\Project\Dataset\dataset"
classes = ['no', 'sphere', 'vort']
image_size = (150, 150)

def load_data(base_dir, classes):
    """Load images and labels into numpy arrays."""
    X, y = [], []
    
    for label, cls in enumerate(classes):
        for set_name in ['train', 'val']:
            cls_path = os.path.join(base_dir, set_name, cls)
            
            if not os.path.exists(cls_path):
                print(f"Path not found: {cls_path}")
                continue
            
            image_files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
            
            for img_file in image_files:
                img_path = os.path.join(cls_path, img_file)
                img = np.load(img_path)

                # Reshape and normalize
                img = img.squeeze()  # Shape: (150, 150)
                img = img / 255.0     # Normalize to [0, 1]

                X.append(img)
                y.append(label)

    # Convert to numpy arrays
    X = np.array(X).reshape(-1, *image_size, 1)
    y = np.array(y)

    return X, y


# Load and split the dataset
X, y = load_data(base_dir, classes)

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y, num_classes=len(classes))

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)

print("Preprocessing complete! ✅")
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
