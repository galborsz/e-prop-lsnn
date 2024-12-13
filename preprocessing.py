import numpy as np
import pickle
from itertools import chain

path = '/Users/giselaalbors/Desktop/university/master/semester 5/e-prop-lsnn/timit_processed/train/'

# Load and process X_train_data
X_train_data = pickle.load(open(path + 'mfccs.pickle', 'rb'))
# Flatten to 2D list (by merging the inner lists)
X_train_data = list(chain.from_iterable(X_train_data))

# Find the min and max values for each feature (column)
min_values = np.min(X_train_data)  # Minimum value for each feature
max_values = np.max(X_train_data)  # Maximum value for each feature

# Apply min-max normalization to scale the features to the range [0, 1]
normalized_X_train = (X_train_data - min_values) / (max_values - min_values)

# Create batches of size 32 for X_train_data
batch_size = 32

X_train_batches = [
    normalized_X_train[i:(i + batch_size)] 
    for i in range(0, len(normalized_X_train), batch_size)
]

with open('mfccs_train_batches.pickle', 'wb') as f:
    pickle.dump(X_train_batches, f)

# Load and process y_train_data
y_train_data = pickle.load(open(path + 'phonems.pickle', 'rb'))
# Flatten to 1D list (by merging the inner lists)
y_train_data = list(chain.from_iterable(y_train_data))

# Create batches of size 32 for y_train_data
y_train_batches = [
    y_train_data[i:(i + batch_size)] 
    for i in range(0, len(y_train_data), batch_size)
]

with open('phonems_train_batches.pickle', 'wb') as f:
    pickle.dump(y_train_batches, f)


path = '/Users/giselaalbors/Desktop/university/master/semester 5/e-prop-lsnn/timit_processed/test/'

# Simulate for each input x^t
X_test_data = pickle.load(open(path + 'mfccs.pickle', 'rb'))
# Flatten to 2D list (by merging the inner lists)
X_test_data = list(chain.from_iterable(X_test_data))

# Find the min and max values for each feature (column)
min_values = np.min(X_test_data)  # Minimum value for each feature
max_values = np.max(X_test_data)  # Maximum value for each feature

# Apply min-max normalization to scale the features to the range [0, 1]
normalized_X_test = (X_test_data - min_values) / (max_values - min_values)

with open('mfccs_test.pickle', 'wb') as f:
    pickle.dump(normalized_X_test, f)

y_test_data = pickle.load(open(path + 'phonems.pickle', 'rb'))
# Flatten to 1D list (by merging the inner lists)
y_test_data = list(chain.from_iterable(y_test_data))

with open('phonems_test.pickle', 'wb') as f:
    pickle.dump(y_test_data, f)

