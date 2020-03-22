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

# Dataset class with premises, hypothesis and label


class Dataset:

    def __init__(self, prem, hyp, lab, max_len):
        self.prem = prem
        self.hyp = hyp
        self.lab = lab
        self.max_len = max_len
        self.dict = {}

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


def encodeData(dataset):
    # Iterator for word
    i_wd = 1

    prem = []
    hyp = []
    lab = []

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
            if wd not in dataset.dict.values():
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
train_ldr = DataLoader(dataset=train_tns, batch_size=batch_size, sampler=train_sampler)

# Vector embedding
embed = nn.Embedding(len(train.dict)+1, 500)

for p, h, l in train_ldr:
    # Going through embedding layer
    vec_prem = embed(p)
    vec_hyp = embed(h)

    # LSTM layer
    input_dim = vec_prem.size()[2]
    hidden_dim = 100
    num_layers = 10
    re_layer = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                       num_layers=num_layers, batch_first=True)
    
    # Creating hidden layer
    hidden_state = torch.randn(num_layers, vec_prem.size()[0], hidden_dim)
    cell_state = torch.randn(num_layers, vec_prem.size()[0], hidden_dim)
    hidden = (hidden_state, cell_state)

    # Putting vectors through recurrent layer
    h_out_x1, hidden_x1 = re_layer(vec_prem, hidden)
    h_out_x2, hidden_x2 = re_layer(vec_hyp, hidden)

    # Concatenating hidden vectors
    cat_tns = torch.cat((hidden_x1[0], hidden_x2[0]), 0)

# Parse test data
test = parseXml("test.xml")
# Convert to integer encoding
encodeData(test)
# Convert to tensors
x1_test_tns = torch.tensor(test.prem)
x2_test_tns = torch.tensor(test.hyp)
y_test_tns = torch.tensor(test.lab)

# Initialize TensorDataset and DataLoader
test_tns = TensorDataset(x1_test_tns, x2_test_tns, y_test_tns)

# Random Sampler (for DataLoader)
test_sampler = SequentialSampler(test_tns)
batch_size = 16
test_ldr = DataLoader(dataset=test_tns, batch_size=batch_size, sampler=test_sampler)

# End of program
print('-----\n', 'Project 1 took', round(time.time() -
                                         start_time, 4), 'seconds to complete.')
