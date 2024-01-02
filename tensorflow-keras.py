#!/usr/bin/env python
# coding: utf-8

# dependencies
import os
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# load data
from datasets import load_dataset

import warnings
import re
warnings.filterwarnings('ignore')


# ----
'''
KerasNLP Training Approach
documentation: https://huggingface.co/datasets/glue

Using the KerasNLP API to train transformers
    import 'GLUE' dataset: General Language Understanding Evaluation benchmark
    import 'CoLA' dataset: Corpus of Linguistic Acceptability
    
'''
# function
dataset = load_dataset('glue', 'cola')
dataset = dataset["train"]

'''loading a tokenizing the data as NumPy arrays
    Notes: 
        - labels consists of lists of 1s and 0s
        - this can be converted to a NumPy array w.o. tokenizing'''

# function
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

'''
setting tokenizer parameters
    Args:
        dataset["sentence"]: The text data to be tokenized, where each element is a string.
        return_tensors="np": Specifies the output format as NumPy arrays.
        padding=True: Ensures all tokenized outputs are of the same length by adding padding.
'''

# function
tokenized_data = tokenizer(
    dataset["sentence"], 
    return_tensors="np", 
    padding=True)

'''
- Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
- "BatchEncoding" here neatly organizes all the information. 
- We can think of it like a container or a table where each row represents a piece of text from your batch, and each column contains the different pieces of information (like token IDs, attention masks) for that text.
'''

# function
tokenized_data = dict(tokenized_data)

# label field is already an array of 0s and 1s
# convert the labels to a NumPy array

# function
labels = np.array(dataset["label"])  

# now we load, compile, and fit the model
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")

# function
# lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5))

# function
model.fit(tokenized_data, labels)

