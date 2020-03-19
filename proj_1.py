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

# Dataset class with premises, hypothesis and label
class Dataset:
    def __init__(self, prem, hyp, lab):
        self.prem = prem
        self.hyp = hyp
        self.lab = lab
        self.dict_wd = {}
        self.dict_lab = {}

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

    # Iterate through all children of root
    for child in root:
        # Get labels
        lab.append(True if child.attrib['value']=='TRUE' else (False if child.attrib['value']=='FALSE' else None))

        # Iterate through all gchildren of root
        for gchild in child:
            if gchild.tag == 't':
                prem.append(gchild.text.split())
            if gchild.tag == 'h':
                hyp.append(gchild.text.split())

    return Dataset(prem, hyp, lab)

# Convert words into integers and record in the dictionary
def encodeData(dataset):
    # Iterator for word
    i_wd = 1;

    prem = []
    hyp = []
    lab = []

    # Iterate through prem
    for p in dataset.prem:
        sentence = []
        for wd in p:
            # Add word to dictionary if not already in there
            if wd not in dataset.dict_wd.values():
                dataset.dict_wd.update({i_wd: wd.casefold()})
                i_wd+=1
            
            # Find key for current word
            for key, val in dataset.dict_wd.items():
                if val == wd.casefold():
                    sentence.append(key)
        prem.append(sentence)

    # Iterate through hyp
    for h in dataset.hyp:
        sentence = []
        for wd in h:
            # Add word to dictionary if not already in there
            if wd not in dataset.dict_wd.values():
                dataset.dict_wd.update({i_wd: wd.casefold()})
                i_wd+=1
            
            # Find key for current word
            for key, val in dataset.dict_wd.items():
                if val == wd.casefold():
                    sentence.append(key)
        hyp.append(sentence)
            
# Driver of the program
start_time = time.time()

# Check if train and test files exist
if not path.exists("train.xml") or not path.exists("test.xml"):
    raise FileNotFoundError("Train and/or test data not found!")

# Parse train data 
train = parseXml("train.xml")
train = encodeData(train)


test = parseXml("test.xml")

# End of program
print('-----\n', 'Project 1 took', round(time.time() -
                                         start_time, 4), 'seconds to complete.')
