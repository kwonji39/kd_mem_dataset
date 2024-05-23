import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pickle_data(data, labels, filename):
    with open(filename, 'wb') as fo:
        pickle.dump({'data': data, 'labels': labels}, fo, protocol=pickle.HIGHEST_PROTOCOL)

data_path = "/home/kwon/cs590_sp24/d/kd_memorization/src/test_dataset/"
batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
train_data = []
train_labels = []

# Load and combine all train batches
for batch_file in batch_files:
    batch_path = data_path + batch_file
    batch = unpickle(batch_path)
    train_data.append(batch[b'data'])
    train_labels.append(batch[b'labels'])

# Convert lists to numpy arrays and stack them
train_data = np.vstack(train_data)
train_labels = np.hstack(train_labels)

# Save the combined dataset to a new file
output_filename = data_path + "train_batch"
pickle_data(train_data, train_labels, output_filename)
