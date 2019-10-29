import pandas as pd
import numpy as np
from keras import models
from keras import layers
from sklearn.utils import shuffle

classes = ['basmati', 'daneboland']
train_data = []
train_targets = []

#   load dataset and labels
for target, c in enumerate(classes):
    df = pd.read_excel(r'{}.xlsx'.format(c), header=None)
    train_data += df.values.tolist()
    train_targets += np.full(len(df), target).tolist()

train_data = np.array(train_data)
train_targets = np.array(train_targets)

train_data, train_targets = shuffle(train_data, train_targets)


# Create function returning a compiled model
def create_model():
    # Start neural network
    model = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    model.add(layers.Dense(units=8, activation='relu', input_shape=(train_data.shape[1],)))

    # # Add fully connected layer with a ReLU activation function
    model.add(layers.Dense(units=8, activation='relu'))

    # Add fully connected layer with a sigmoid activation function
    model.add(layers.Dense(units=1, activation='sigmoid'))

    # Compile neural network
    model.compile(loss='binary_crossentropy',  # Cross-entropy
                  optimizer='rmsprop',  # Root Mean Square Propagation
                  metrics=['accuracy'])  # Accuracy performance metric

    # Return compiled model
    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_accs = []

for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
        
    model = create_model()
    # Train the model 
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=40)
    # Evaluate the model on the validation data
    score, acc = model.evaluate(val_data, val_targets)
    all_accs.append(acc)

acc = np.mean(all_accs)
print(acc)
model.save('rice_detection.h5')
