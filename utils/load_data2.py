import numpy as np
import os
from PIL import Image
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf

def load_train(datadir):
    """
    Load images and corresponding SMILES strings from the dataset directory.

    Args:
        datadir (str): Directory containing subfolders with images and SMILES files.

    Returns:
        tuple: Tuple containing numpy arrays of images (x) and SMILES strings (y).
    """
    x = []
    y = []

    for sub_folder in os.listdir(datadir):
        sub_folder_path = os.path.join(datadir, sub_folder)
        if not os.path.isdir(sub_folder_path):
            continue  # Skip non-directory files

        for file_name in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, file_name)

            # Load images
            if file_name.endswith(".png"):
                img = Image.open(file_path).convert("L")  # Convert to grayscale
                img = img.resize((600, 420))  # Resize to 600x420
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0-1
                img_rgb = np.stack([img_array] * 3, axis=-1)  # Create an RGB representation
                x.append(img_rgb)

            # Load SMILES strings
            elif file_name == "smiles.txt":
                with open(file_path, "r") as smiles_file:
                    y.append(smiles_file.read().strip())  # Add SMILES string to list

    # Convert image list to numpy array
    x = np.array(x, dtype=np.float32)

    return x, y


def prepare_dataset(x, y, max_smile_sequence_length, training_batch_size):
    """
    Prepare a TensorFlow dataset for training.

    Args:
        x (numpy.ndarray): Array of images.
        y (list): List of SMILES strings.
        max_smile_sequence_length (int): Maximum length for SMILES sequences.
        training_batch_size (int): Batch size for training.

    Returns:
        tuple: Tuple containing the prepared dataset and the TextVectorization layer.
    """
    # Define a TextVectorization layer
    vectorizer = TextVectorization(
        max_tokens=76,  # SMILES notation uses up to 76 unique characters
        output_sequence_length=max_smile_sequence_length,
        standardize=None  # SMILES strings are case-sensitive; no additional standardization
    )
    # Adapt the vectorizer to the SMILES vocabulary
    vectorizer.adapt(y)

    # Tokenize the SMILES strings
    y_tokenized = vectorizer(y)

    # Convert tokenized SMILES to TensorFlow tensors
    y_tokenized = tf.convert_to_tensor(y_tokenized)

    # Create a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y_tokenized))
    dataset = dataset.shuffle(buffer_size=len(x)).batch(training_batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, vectorizer


if __name__ == "__main__":
    # Load training data
    x, y = load_train("./dataset")

    # Prepare the dataset
    max_smile_sequence_length = 512
    training_batch_size = 32
    dataset, vectorizer = prepare_dataset(x, y, max_smile_sequence_length, training_batch_size)

    # Print dataset details
    print("Length of x_train:", len(x))
    print("Length of y_train:", len(y))
    print("Dataset example:", next(iter(dataset)))