import numpy as np 
import os
from PIL import Image
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder 

def load_train(datadir):

    x = []
    y = []

    for sub_folder in os.listdir(datadir):
        for file_name in os.listdir(f"{datadir}/{sub_folder}"):
            
            if file_name.endswith(".png"):
                img = Image.open(f"{datadir}/{sub_folder}/{file_name}").convert("L")
                # Normalize image in range 0...1 and add to train set
                img = img.resize((600,420))
                x.append(np.array(img)/255) 

            elif file_name == "smiles.txt":
                with open(f"{datadir}/{sub_folder}/{file_name}", "r") as smiles:
                    y.append(smiles.read())

    x = np.array(x, dtype=np.bool)

    encoder = OneHotEncoder(sparse_output=False)
    smiles_array = np.array(y).reshape(-1, 1)
    encoder.fit(smiles_array)

    y = encoder.transform(np.array(y).reshape(-1,1))
    return x,y 



def prepare_dataset(x, y, max_smile_sequence_length, training_batch_size):
    
    """
    max_smile_sequence_len defines how long each individual input sequence is. It determines how much padding or truncation happens to each sample.
    training_batch_size defines how many sequences are processed together in each training step.
    """

    vectorizer = TextVectorization(
        max_tokens=76,  #https://chemistry.stackexchange.com/questions/158969/how-many-characters-does-the-smiles-notation-uses
        output_sequence_length=max_smile_sequence_length,
        standardize=None
    )
    vectorizer.adapt(y)  # Learn the vocabulary from the SMILES strings
    y_tokenized = vectorizer(y)

    # Create a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y_tokenized))
    #dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, vectorizer


if __name__=="__main__":
    x,y = load_train("./dataset")

    dataset = prepare_dataset(x,y,512,32 )
    print("length of x_train",len(x), "\n length of y_train", len(y))

    print(dataset)