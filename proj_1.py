'''
Name: Kevin Chen
NetID: nkc160130
CS 6320
Due: 3/30/2020
Dr. Moldovan
Version: Python 3.8.0
'''

import time
import xml.etree.ElementTree as et
import os.path as path
import sys
import string

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dataset class with premises, hypothesis and label


class Dataset:

    def __init__(self, prem, hyp, lab, max_len):
        self.prem = prem
        self.hyp = hyp
        self.lab = lab
        self.max_len = max_len
        self.dict = {}

# Network class with layers


class NNetwork(nn.Module):
    def __init__(self, dict_size, device):
        super().__init__()
        # Vector embedding
        self.embed = nn.Embedding(dict_size+1, 500)
        # Recurrent layer
        self.lstm = nn.LSTM(input_size=500, hidden_size=256,
                            num_layers=15, batch_first=True)
        # Fully connected layer
        self.fcl = nn.Linear(512, 2)
        self.softmax = nn.Softmax()
        self.device = device

    def forward(self, tns1, tns2):
        # Pass the input tensor through each operation
        tns1 = tns1.to(self.device)
        tns2 = tns2.to(self.device)

        # Embed word tensor to vector
        tns1 = self.embed(tns1)
        tns2 = self.embed(tns2)

        # Creating hidden layer
        hidden_state = torch.randn(15, tns1.size()[0], 256).to(self.device)
        cell_state = torch.randn(15, tns1.size()[0], 256).to(self.device)
        hidden = (hidden_state, cell_state)
        # LSTM layer
        out1, hidden1 = self.lstm(tns1, hidden)
        tns1 = out1.squeeze()[:, -1]

        # Creating hidden layer
        hidden_state = torch.randn(15, tns2.size()[0], 256).to(self.device)
        cell_state = torch.randn(15, tns2.size()[0], 256).to(self.device)
        hidden = (hidden_state, cell_state)
        # LSTM layer
        out2, hidden2 = self.lstm(tns2, hidden)
        tns2 = out2.squeeze()[:, -1]

        # Concatenate the two input tensors
        tns = torch.cat((tns1, tns2), 1)

        # Fully connected layer
        tns = self.fcl(tns)
        tns = torch.softmax(tns, dim=1)

        return tns

# Function to parse XML file and extract premise, hypothesis and label data. Returns Dataset object.


def parseXml(xml_file):
    # Check if xml file has correct formatting (will throw exception if not)
    et.fromstring(open(xml_file, 'r').read())

    # Parse xml file into an ElementTree object
    xml_tree = et.parse(xml_file)
    root = xml_tree.getroot()

    # Initialize properties of Dataset
    prem = []
    hyp = []
    lab = []
    max_len = 0

    # Iterate through all children of root
    for child in root:
        # Get labels
        lab.append(True if child.attrib['value'] == 'TRUE' else (
            False if child.attrib['value'] == 'FALSE' else None))

        # Iterate through all gchildren of root
        for gchild in child:
            input_child = gchild.text.casefold().translate(
                str.maketrans('', '', string.punctuation)).split()
            if len(input_child) > max_len:
                max_len = len(input_child)

            if gchild.tag == 't':
                prem.append(input_child)
            if gchild.tag == 'h':
                hyp.append(input_child)

    return Dataset(prem, hyp, lab, max_len)

# Convert words into integers and record in the dictionary


def encodeData(dataset, dict=None):
    # Iterator for word
    i_wd = 1

    prem = []
    hyp = []
    lab = []

    # Dictionary passed in (test set)
    if dict is not None:
        dataset.dict = dict.copy()

    # Iterate through prem
    for p in dataset.prem:
        sentence = []
        for wd in p:
            # Add word to dictionary if not already in there
            if wd not in dataset.dict.values():
                dataset.dict.update({i_wd: wd})
                i_wd += 1

            # Find key for current word
            for key, val in dataset.dict.items():
                if val == wd:
                    sentence.append(key)
        prem.append(sentence)

    # Iterate through hyp
    for h in dataset.hyp:
        sentence = []
        for wd in h:
            # Add word to dictionary if not already in there
            if wd not in dataset.dict.values() and dict is None:
                dataset.dict.update({i_wd: wd.casefold()})
                i_wd += 1

            # Find key for current word
            for key, val in dataset.dict.items():
                if val == wd.casefold():
                    sentence.append(key)



        hyp.append(sentence)

    # Iterate through lab
    for l in dataset.lab:
        if l == True:
            lab.append(2)
        if l == False:
            lab.append(1)

    max_len = dataset.max_len

    # See if maxlength is set
    if len(sys.argv) == 2:  # Maxlength is set
        max_len = int(sys.argv[1])

    # Appending zeros to make all vectors uniformely large
    for p in prem:
        while len(p) < max_len:
            p.append(0)
    for h in hyp:
        while len(h) < max_len:
            h.append(0)

    dataset.prem = prem
    dataset.hyp = hyp
    dataset.lab = lab


# Driver of the program
start_time = time.time()
torch.manual_seed(0)

# Try to use GPU for PyTorch, if available; otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Check if train and test files exist
if not path.exists("train.xml") or not path.exists("test.xml"):
    raise FileNotFoundError("Train and/or test data not found!")

# Parse train data
train = parseXml("train.xml")
# Convert to integer encoding
encodeData(train)
# Convert to tensors
x1_train_tns = torch.tensor(train.prem)
x2_train_tns = torch.tensor(train.hyp)
y_train_tns = torch.tensor(train.lab)

# Initialize TensorDataset and DataLoader
train_tns = TensorDataset(x1_train_tns, x2_train_tns, y_train_tns)

# Random Sampler (for DataLoader)
train_sampler = RandomSampler(train_tns)
batch_size = 16
train_ldr = DataLoader(
    dataset=train_tns, batch_size=batch_size, sampler=train_sampler)

model = NNetwork(len(train.dict), device).to(device)

# Define hyperparameters
n_epochs = 20

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Put prem and hyp through network for training
for p, h, l in train_ldr:

    # Training Run
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch

        output = model(p, h).to(device)

        # Convert to indexes to calculate loss
        lab_compare = []
        for la in l:
            if la == 1:
                lab_compare.append(0)
            if la == 2:
                lab_compare.append(1)
        lab_compare = torch.tensor(lab_compare).to(device)

        # calculating loss
        loss = criterion(output, lab_compare)
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly

        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))


# Parse test data
test = parseXml("test.xml")
# Convert to integer encoding
encodeData(test, train.dict)
# Convert to tensors
x1_test_tns = torch.tensor(test.prem)
x2_test_tns = torch.tensor(test.hyp)
y_test_tns = torch.tensor(test.lab)

# Initialize TensorDataset and DataLoader
test_tns = TensorDataset(x1_test_tns, x2_test_tns, y_test_tns)

# Random Sampler (for DataLoader)
test_sampler = SequentialSampler(test_tns)
batch_size = 16
test_ldr = DataLoader(
    dataset=test_tns, batch_size=batch_size, sampler=test_sampler)

# Put prem and hyp through network for test

for p, h, l in train_ldr:
    inference_time = time.time()
    calculated = model(p, h).cpu()
    print('---------------')
    print('Throughput: ', round(time.time() - inference_time, 4), 'seconds')
    print('---------------')

    l_cal = []
    for cal in calculated:
        if cal[0] > cal[1]:
            l_cal.append(1)
        else:
            l_cal.append(2)

    # Report performance
    print('Predicted:\t', l_cal)
    print('Actual:\t\t', l.tolist())
    print('---------------')
    print('| Performance |')
    print('---------------')
    print('Accuracy score: ', accuracy_score(l_cal, l.tolist()))
    print('Precision score: ', precision_score(l_cal, l.tolist()))
    print('Recall score: ', recall_score(l_cal, l.tolist()))
    print('F1 score: ', f1_score(l_cal, l.tolist()))
    print('---------------\n')

# End of program
print('-----\n', 'Project 1 took', round(time.time() -
                                         start_time, 4), 'seconds to complete.')
