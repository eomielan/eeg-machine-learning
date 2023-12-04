import pandas as pd
from glob import glob
import os
import numpy as np
from keras.layers import LSTM, Dense, Dropout, Bidirectional

from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import tensorflow as tf
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import timesynth as ts





def Maze():
    maze = "/Users/ishwarjotgrewal/Downloads/465Data/Maze"

    # Use glob to get a list of all CSV files in the directory
    csv_files = glob(f"{maze}/*.csv")
    image_folder = os.path.join(maze, 'Images')
    os.makedirs(image_folder, exist_ok=True)

    # Create subfolders for 'maze' and 'recall' images
    maze_folder = os.path.join(image_folder, 'maze')
    recall_folder = os.path.join(image_folder, 'recall')
    os.makedirs(maze_folder, exist_ok=True)
    os.makedirs(recall_folder, exist_ok=True)

    # Initialize an empty list to store DataFrames
    dataframes_list = []

    # Loop through each CSV file and read it into a DataFrame
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Drop the first row
        df = df.iloc[1:]
        
        
        df["Delta"] = df.iloc[:, 1:5].mean(axis=1)
        df["Theta"] = df.iloc[:, 5:9].mean(axis=1)
        df["Alpha"] = df.iloc[:, 9:13].mean(axis=1)
        df["Beta"] = df.iloc[:, 13:17].mean(axis=1)
        df["Gamma"] = df.iloc[:, 17:21].mean(axis=1)

        dataframes_list.append(df)

    # List of column names to include in the new DataFrames
    selected_columns = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    # Create a new list to store DataFrames with selected columns
    mazes = []

    # Iterate through the original list of DataFrames and select specific columns
    for i, df in enumerate(dataframes_list):
        file_name = os.path.basename(csv_files[i])
        
        # Create a new DataFrame with selected columns
        selected_df = df[selected_columns]
        
        # Name the new DataFrame based on the original file name
        selected_df_name = f"{file_name}_selected"
        
        # Add the new DataFrame to the list with the specified name
        mazes.append(selected_df.dropna())

    min_rows_mazes = min(df.shape[0] for df in mazes)

    # Trim the ends of each DataFrame in 'mazes' to match the length of the smallest DataFrame
    for i, df in enumerate(mazes):
        mazes[i] = df.iloc[:min_rows_mazes] 

    return mazes


def Recall():
    recall = "/Users/ishwarjotgrewal/Downloads/465Data/Recall"

    # Use glob to get a list of all CSV files in the directory
    csv_files = glob(f"{recall}/*.csv")
    image_folder = os.path.join(recall, 'Images')
    os.makedirs(image_folder, exist_ok=True)

    # Create subfolders for 'maze' and 'recall' images
    maze_folder = os.path.join(image_folder, 'maze')
    recall_folder = os.path.join(image_folder, 'recall')
    os.makedirs(maze_folder, exist_ok=True)
    os.makedirs(recall_folder, exist_ok=True)

    # Initialize an empty list to store DataFrames
    dataframes_list = []

    # Loop through each CSV file and read it into a DataFrame
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Drop the first row
        df = df.iloc[1:]
        
        df["Delta"] = df.iloc[:, 1:5].mean(axis=1)
        df["Theta"] = df.iloc[:, 5:9].mean(axis=1)
        df["Alpha"] = df.iloc[:, 9:13].mean(axis=1)
        df["Beta"] = df.iloc[:, 13:17].mean(axis=1)
        df["Gamma"] = df.iloc[:, 17:21].mean(axis=1)

        dataframes_list.append(df)

    min_rows = 15235
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.iloc[:min_rows]

    # List of column names to include in the new DataFrames
    selected_columns = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    # Create a new list to store DataFrames with selected columns
    recalls = []

    # Iterate through the original list of DataFrames and select specific columns
    for i, df in enumerate(dataframes_list):
        file_name = os.path.basename(csv_files[i])
        
        # Create a new DataFrame with selected columns
        selected_df = df[selected_columns]
        
        # Name the new DataFrame based on the original file name
        selected_df_name = f"{file_name}_selected"
        
        # Add the new DataFrame to the list with the specified name
        recalls.append(selected_df.dropna())

    min_rows_recalls = min(df.shape[0] for df in recalls)

    # Trim the ends of each DataFrame in 'recalls' to match the length of the smallest DataFrame
    for i, df in enumerate(recalls):
        recalls[i] = df.iloc[:min_rows_recalls] 

    return recalls


maze_list = Maze()
recall_list = Recall()

# Trimming
min_rows = min(min(df.shape[0] for df in maze_list), min(df.shape[0] for df in recall_list))
maze_list = [df.iloc[:min_rows] for df in maze_list]
recall_list = [df.iloc[:min_rows] for df in recall_list]

# Augmenting Maze data
augmented_data_maze = []

for series in maze_list:
    time_sampler = ts.TimeSampler(stop_time=1)
    regular_time_samples = time_sampler.sample_regular_time(num_points=len(series))
    
    signal = ts.signals.Sinusoidal(frequency=0.25)
    noise = ts.noise.GaussianNoise(std=0.3)
    
    timeseries = ts.TimeSeries(signal, noise_generator=noise)
    
    samples, _, _ = timeseries.sample(regular_time_samples)
    
    augmented_data_maze.append(samples)

# Convert the list to a numpy array
augmented_data_maze = np.array(augmented_data_maze)

# Concatenate the original and augmented data for the 'maze' class
maze3d = np.stack([df.values for df in maze_list])
maze3d_augmented = np.concatenate([maze3d, augmented_data_maze[:, :, np.newaxis]], axis=2)


# Combine the datasets and labels for 'maze' class
X_maze = np.concatenate([maze3d_augmented, maze3d_augmented])
y_maze = np.concatenate([np.zeros(maze3d_augmented.shape[0]), np.zeros(augmented_data_maze.shape[0])])

# Augmenting Recall data
augmented_data_recall = []

for series in recall_list:
    time_sampler = ts.TimeSampler(stop_time=1)
    regular_time_samples = time_sampler.sample_regular_time(num_points=len(series))
    
    signal = ts.signals.Sinusoidal(frequency=0.25)
    noise = ts.noise.GaussianNoise(std=0.3)
    
    timeseries = ts.TimeSeries(signal, noise_generator=noise)
    
    samples, _, _ = timeseries.sample(regular_time_samples)
    
    augmented_data_recall.append(samples)

# Convert the list to a numpy array
augmented_data_recall = np.array(augmented_data_recall)

# Concatenate the original and augmented data for the 'recall' class
recall3d = np.stack([df.values for df in recall_list])
recall3d_augmented = np.concatenate([recall3d, augmented_data_recall[:, :, np.newaxis]], axis=2)


# Combine the datasets and labels for 'recall' class
X_recall = np.concatenate([recall3d_augmented, recall3d_augmented])
y_recall = np.concatenate([np.ones(recall3d_augmented.shape[0]), np.ones(augmented_data_recall.shape[0])])

# Combine 'maze' and 'recall' datasets
X = np.concatenate([X_maze, X_recall])
y = np.concatenate([y_maze, y_recall])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data using the same scaler for both training and testing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2]))

# Reshape back to 3D
X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)

model = Sequential()
model.add(Bidirectional(LSTM(units=64, activation='tanh', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))))
model.add(Dropout(0.5))  # Adjust the dropout rate
model.add(Dense(64, activation='relu'))  # Additional dense layer with more units
model.add(Dropout(0.5))  # Adjust the dropout rate
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')