# system tools
import os
import sys
sys.path.append("../../../../cds-lang-repo/cds-language")

# data munging tools
import pandas as pd
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Visualisation
import matplotlib.pyplot as plt

def vectorizerobj(): # creating a vectorizer object with the following parameters:
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams
                                lowercase =  True,       # make everything lowercase
                                max_df = 0.95,           # remove 5% most common words
                                min_df = 0.05,           # remove 5% rarest words
                                max_features = 500)      # keep top 500 features