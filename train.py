import numpy as np
import pandas as pd
import math
from ScratchTorch.nn import MLP,ClassifcationNN
from ScratchTorch.grad import Tensor
import pickle

# loading the dataset

print("Loading dataset...")

data = pd.read_csv("dataset/train.csv")

colmns = data.columns.to_list()
colmns.remove('label')

# Pull out the pixel columns and scale values to the 0-1 range so gradients behave better.
features = data[colmns].to_numpy() / 255

# Keep the raw digit labels around before we split anything.
labels = data['label'].to_numpy()

nin = len(features[0]) # num of inputs to neural network
# features = features[:500] # for testing purpose the training is being done on 500 samples after optimisation of ScratchTorch library i will try on full dataset.

# Shuffle the rows once so the split is random but still reproducible.
rng = np.random.default_rng(150)
indices = rng.permutation(len(features))

# using a 80:20 split
split_index = int(0.8 * len(features))
train_indices = indices[:split_index]
test_indices = indices[split_index:]

# Build the two datasets from the shuffled row indices.
train_data = features[train_indices]
test_data = features[test_indices]

# Split the labels alongside the images so they stay aligned.
train_label = labels[train_indices]
test_label = labels[test_indices]

# Convert class numbers into one-hot vectors for softmax training.
y_train = np.eye(10)[train_label]
test_labels = np.eye(10)[test_label]


# model

net = ClassifcationNN(
    784,
    [64, 64, 10]
)

# hyperparameter

epochs = 30
batch_size =128

# cosine learning decay
initial_learning_rate = 0.1
min_learning_rate = .035



def accuracy(model, x_data, y_labels, eval_batch_size=512):
    correct = 0

    for start in range(0, len(x_data), eval_batch_size):
        end = start + eval_batch_size
        xb = x_data[start:end]
        yb = y_labels[start:end]

        probs = model(Tensor(xb))
        predictions = np.argmax(probs.data, axis=1)
        correct += np.sum(predictions == yb)

    return correct / len(x_data)


# training loop

learning_rate = initial_learning_rate
for epoch in range(epochs):
    permutation = rng.permutation(len(train_data))

    epoch_loss = 0.0
    num_batches = 0

    for start in range(0, len(train_data), batch_size):

        end = start + batch_size

        batch_idx = permutation[start:end]

        xb = train_data[batch_idx]
        yb = y_train[batch_idx]

        # forward
        
        x = Tensor(xb)

        probs = net(x)

        y_true = Tensor(yb)

        
        loss = -(y_true * probs.log()).sum()
        loss = loss / len(xb)

        # reset the gradients implemented in ScratchTorch.
        loss.reset_gradients()

        # backward pass which is implemented in ScratchTorch
        loss.backward()
        
        # TODO: to implement this as a function in neural network class.
        for p in net.parameters():
            p.data -= learning_rate * p.grad

        epoch_loss += float(loss.data)
        num_batches += 1
        
    avg_loss = epoch_loss / num_batches
    train_accuracy = accuracy(net, train_data, train_label)
    test_accuracy = accuracy(net, test_data, test_label)
    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Test Acc: {test_accuracy * 100:.2f}% |"
        f"Learning Rate: {learning_rate}"
    )
    # learning rate decay so we dont overfit over the data.
    # cosine Decay
    # lr = lr_min + 0.5 * (lr_initial - lr_min) * (1 + cos((T_cur / T_max) * pi))
    learning_rate = initial_learning_rate +(.5*(initial_learning_rate - min_learning_rate))*(1 + math.cos((epoch+1/epochs)))
    


final_train_accuracy = accuracy(net, train_data, train_label)
final_test_accuracy = accuracy(net, test_data, test_label)

print("\nFinal evaluation")
print(f"Train Accuracy: {final_train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {final_test_accuracy * 100:.2f}%")

