# General Imports
import pickle as pkl
import numpy as np
import pandas as pd
import os
import warnings

# Domain Imports
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, LSTM, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from IPython.display import clear_output

# Local Package Imports
from build_datasets.build_datasets import dataset_builder
import gcloud_helper as gc

# Turn Off Warnings
warnings.filterwarnings("ignore")

# Load the dataset
with open('data/datasets/x_train.pkl', 'rb') as fpath:
    x_train = pkl.load(fpath)

with open('data/datasets/y_train.pkl', 'rb') as fpath:
    y_train = pkl.load(fpath)

# Preprocess the data
NN_pipe = dataset_builder().ml_pipe(model=None)
x_train = NN_pipe.fit_transform(x_train)

# Convert data to numpy arrays
x_train = np.array(x_train, dtype=np.float32)
y_train_encoded = np.array(y_train, dtype=np.int32)

# Split data into training and validation sets
x_train, x_val, y_train_encoded, y_val = train_test_split(x_train, y_train_encoded, test_size=0.2, random_state=42)

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore the best weights after stopping
)

# Model architectures to test
architectures = [
    [32],              # Small model
    [64],              # Small model
    [32, 64],          # Medium model
    [16, 32, 64],      # Larger model
    [8, 16, 32, 64],   # Large model
    [64, 32],          # Medium model reversed
    [64, 32, 16],      # Larger model reversed
    [64, 32, 16, 8],   # Large model reversed
]

# Additional types of architectures:
additional_architectures = [
    {'type': 'CNN', 'filters': [32, 64], 'kernel_size': 3, 'pool_size': 2},  # CNN model
    {'type': 'RNN', 'units': [128, 64], 'cell_type': 'LSTM'},  # LSTM-based RNN
]

# Initialize best model variables
best_model = None
best_val_loss = float('inf')

# Directory to save the best model
save_dir = 'data/models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop over architectures
for arch in architectures:
    print(f'Training Model with Architecture: {arch}')
    # Define the model
    model = Sequential()
    model.add(Dense(arch[0], input_shape=(120,), activation='relu', kernel_regularizer=l2(0.01)))
    for units in arch[1:]:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(12, activation='softmax'))  # Output layer
    
    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['sparse_categorical_crossentropy']
    )

    # Fit the model
    history = model.fit(x_train, y_train_encoded, validation_data=(x_val, y_val),
                        epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

    # Check if this model is the best based on validation loss
    val_loss = min(history.history['val_loss'])
    print(f'Model with architecture {arch} achieved validation loss: {val_loss}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_architecture = arch

# Loop over additional architectures (CNNs, RNNs, etc.)
for arch in additional_architectures:
    print(f'Training Model with Architecture: {arch}')
    model = Sequential()
    if arch['type'] == 'CNN':
        model.add(Conv1D(arch['filters'][0], arch['kernel_size'], activation='relu', input_shape=(120, 1)))
        model.add(MaxPooling1D(arch['pool_size']))
        model.add(Conv1D(arch['filters'][1], arch['kernel_size'], activation='relu'))
        model.add(MaxPooling1D(arch['pool_size']))
        model.add(Flatten())
    elif arch['type'] == 'RNN':
        if arch['cell_type'] == 'LSTM':
            model.add(LSTM(arch['units'][0], input_shape=(120, 1), return_sequences=True))
            model.add(LSTM(arch['units'][1]))
    
    model.add(Dense(12, activation='softmax'))  # Output layer
    
    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['sparse_categorical_crossentropy']
    )

    # Reshape data for CNN and RNN models
    if arch['type'] in ['CNN', 'RNN']:
        x_train_reshaped = np.expand_dims(x_train, axis=-1)  # Reshape for CNN/RNN
        x_val_reshaped = np.expand_dims(x_val, axis=-1)
    else:
        x_train_reshaped = x_train
        x_val_reshaped = x_val

    # Fit the model
    history = model.fit(x_train_reshaped, y_train_encoded, validation_data=(x_val_reshaped, y_val),
                        epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

    # Check if this model is the best based on validation loss
    val_loss = min(history.history['val_loss'])
    print(f'{arch["type"]} model with architecture {arch} achieved validation loss: {val_loss}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_architecture = arch

# Save the best model
if best_model is not None:
    best_model.save(f'{save_dir}/best_model_{best_architecture}.keras')
    print(f'Best model saved with architecture: {best_architecture}')
else:
    print('No model was trained.')































# # General Imports
# import pickle as pkl
# import numpy as np
# import pandas as pd
# import warnings

# # Domain Imports
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from xgboost import XGBClassifier
# from sklearn.exceptions import ConvergenceWarning
# from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.metrics import log_loss
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from IPython.display import clear_output

# # Local Package Imports
# from build_datasets.build_datasets import dataset_builder
# import gcloud_helper as gc

# # Turn Off Warnings
# warnings.filterwarnings("ignore")


# with open('data/datasets/x_train.pkl', 'rb') as fpath:
#     x_train = pkl.load(fpath)

# with open('data/datasets/y_train.pkl', 'rb') as fpath:
#     y_train = pkl.load(fpath)

# NN_pipe = dataset_builder().ml_pipe(model=None)
# x_train = NN_pipe.fit_transform(x_train)

# #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# # # Ensure data is in the correct format
# # x_train = np.array(x_train, dtype=np.float32)
# y_train_encoded = np.array(y_train, dtype=np.int32)

# # Define the model
# model = Sequential([
#     Dense(64, input_shape=(120,), activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(12, activation='softmax')
# ])

# early_stopping = EarlyStopping(
#     monitor='val_loss',       # Metric to monitor (e.g., validation loss)
#     patience=15,              # Number of epochs with no improvement before stopping
#     restore_best_weights=True # Revert to the best weights at the end
# ) 

# # Compile the model
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer='adam',
#     metrics=['sparse_categorical_crossentropy']
# )

# # Fit the model
# model.fit(x_train, y_train_encoded, validation_split=0.2, epochs=100, batch_size=32, 
#           callbacks=[early_stopping])

# model.save('data/models/NN_64_32.keras')