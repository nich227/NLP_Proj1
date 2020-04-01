# Project 1: Recognizing Textual Entailment

## Description
For Project-1, you will implement a deep learning model that recognizes the textual entailment relation between two sentences.
Here, we are given two sentences: the premise, denoted by the letter t and the hypothesis, denoted by the letter h. We say
that the premise entails the hypothesis (i.e. t &#8594; h if the meaning of h can be inferred from the meaning of t). 

The task of textual entailment is set up as a binary classication problem where, given the premise and the hypothesis, the
goal is to classify the relation between them as _Entails_ or _Not Entails_.

## Instructions
1. Download the train.xml and test.xml files.
2. Install dependencies:

    _Linux or macOS_
    ```bash
    pip3 install -r requirements.txt
    ```

    _Windows_
    ```bash
    pip install -r requirements.txt
    ```

3. To run, type in the command line interpreter:

    _Linux or macOS_
    ```bash
    python3 proj_1.py 
    ```

    _Windows_
    ```bash
    python proj_1.py
    ```

## Links
Colab URL: https://colab.research.google.com/drive/1cLAHRpsEGvtqWSD6NSwaH3UG-cqk_8Lx
