#### PACKAGE IMPORTS ####
import tensorflow as tf
import glob2
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

# split a univariate sequence into samples

def split_sequence(df, window_size, nonoverlap_size):
    '''Takes df of 8 channels of raw EMG data from Myo armband. window_size and nonoverlap_size indicating the rolling window paradigm'''
    X, y = list(), list()

    for i in np.arange(0, len(df), step = nonoverlap_size):
        # find the end of this pattern
        end_ix = i + window_size
        # check if we are beyond the sequence
        if end_ix > len(df)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = df.iloc[i:end_ix, :8], df.iloc[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

def get_CNN_model(input_shape, output_shape, wd, dr = 0.3):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), 
                     input_shape = (input_shape[1], input_shape[2], 1), activation = 'relu', 
                     padding = 'same', name = 'conv_1_input', kernel_regularizer = tf.keras.regularizers.l2(wd)))
#     model.add(Dropout(dr, name = 'Dropout_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'Conv_2', kernel_regularizer = tf.keras.regularizers.l2(wd)))
#     model.add(Dropout(dr, name = 'Dropout_2'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', name = 'Conv_3', kernel_regularizer = tf.keras.regularizers.l2(wd)))
#     model.add(Dropout(dr, name = 'Dropout_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name = 'Maxpool_1'))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu', name = 'Dense_1', kernel_regularizer = tf.keras.regularizers.l2(wd)))
    model.add(Dropout(dr, name = 'Dropout_1'))
    model.add(Dense(output_shape, activation = 'softmax', name = 'out_layer'))
    
    return model

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__



def mirror_sensor(df_unmirrored):
    A_values = df_unmirrored.iloc[:, :8].values
    A_mirrored = pd.DataFrame(A_values, columns=['c7', 'c6', 'c5', 'c4', 'c3', 'c2', 'c1', 'c8'])
    A_mirrored = A_mirrored[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']]
    A_mirrored = pd.concat((A_mirrored, df_unmirrored.iloc[:, 8:].reset_index(drop = True)), axis = 1)

    return A_mirrored
def train_CNN_model(emg_list, label_list, side_list, ID):

    df = pd.DataFrame(np.array(emg_list).reshape(-1, 8), columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c7', 'c7', 'c8'])
    df['side'] = np.array(side_list).reshape(-1, 1)
    df['Gesture'] = np.array(label_list).reshape(-1, 1)

    if not os.path.exists(f'{ID}'):
        os.mkdir(f'{ID}')

    filename = glob2.glob(f'{ID}/emgRawData*')
    if len(filename) == 0:
        df.to_pickle(f'{ID}/emgRawData_1.pkl')
    else:
        session = int(sorted(filename, key=os.path.getmtime)[-1].split('_')[1].split('.')[0])
        df.to_pickle(f'{ID}/emgRawData_{session + 1}.pkl')

    # Training the model
    df_all = pd.DataFrame(None)
    for file in filename[-5:]:
        df_all = pd.concat((df_all, pd.read_pickle(file)), axis = 0)

    # Saving the gestureProfile
    df_healthy = df_all[df_all['side'] == 'healthy']
    df_healthy = mirror_sensor(df_healthy)

    gestureProfile = []
    for gesture in df_healthy['Gesture'].unique():
        A = df_healthy[df_healthy['Gesture'] == gesture].iloc[:, :8]
        gestureProfile.append({'Gesture': gesture,
                               'RMS': np.sqrt(np.mean(A ** 2, axis=0)).values})
    gestureProfile = pd.DataFrame(gestureProfile)
    gestureProfile.to_pickle(f'{ID}/gestureProfile_{ID}.pkl')

    df_training = df_all[df_all['side'] == 'training']
    df_training.reset_index(inplace=True, drop=True)
    df_temp = df_training.groupby('Gesture').head(200)

    df_training.loc[df_temp.index, :] = np.nan
    df_training.dropna(inplace=True)

    X_train, y_train = split_sequence(df_training, window_size = 30, nonoverlap_size = 1)
    X_train = X_train.reshape((-1, X_train.shape[1], X_train.shape[2], 1))
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_train = to_categorical(y_train)

    reduce_lr = ReduceLROnPlateau(monitor='acc', patience=3, mode='max', verbose=0)
    early_stopping = EarlyStopping(monitor='acc', patience=5, mode='max', verbose=0)

    model = get_CNN_model(X_train.shape, y_train.shape[1], wd = 0.01)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')])

    model.fit(X_train, y_train,
              callbacks=[reduce_lr, early_stopping],
              epochs=30, verbose=1, batch_size = 256)

    # Run the function
    make_keras_picklable()

    # Save
    with open(f'{ID}/model.pkl', 'wb') as f:
        pickle.dump(model, f)
