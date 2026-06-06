from micrograd.nn import MLP
import pandas as pd
import numpy as np


data = pd.read_csv("dataset/train.csv")
colmns = data.columns.to_list()
colmns.remove('label')

# Pull out the pixel columns and scale values to the 0-1 range so gradients behave better.
features = data[colmns].to_numpy() / 255

# Keep the raw digit labels around before we split anything.
labels = data['label'].to_numpy()

nin = len(features[0]) # num of inputs to neural network
features = features[:500] # for testing purpose the training is being done on 500 samples after optimisation of micrograd library i will try on full dataset.

# Shuffle the rows once so the split is random but still reproducible.
rng = np.random.default_rng(150)
indices = rng.permutation(len(features))

# Use an 80/20 split: most examples for learning, the rest for honest evaluation.
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


neural_network = MLP(nin, [64,64,10])

# training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"--epoch number:{epoch} started--")
    ypred = []
    loss = []
    # forward pass
    idx = 0
    for x,y  in zip(train_data,train_label):
        if len(train_data) < 100:
            print(f"    --sample number:{idx+1}--")
        else:
            if idx%10 == 0:
                print(f"    --sample number:{idx+1}--")

        pred = neural_network(x)
        ypred.append(pred)
        # finding loss for each sample and appending it to the loss function
        loss.append(sum((y*class_prob.exp() for class_prob in pred)))
        idx+=1
        
    
    # computing the total loss over the dataset
    loss = sum(loss)
    
    # backward pass
    for param in neural_network.parameters():
        param.grad = 0.0 #resetting the gradient
    loss.backward() #calculating the gradients for each parameter
    
    for param in neural_network.parameters():
        param.data+= -.01 * param.grad #updating the weights
    
    
    print(f"--epoch number:{epoch} completed. loss={loss}--")
        
    
    