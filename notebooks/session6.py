import gensim
import gensim.downloader
import gensim.downloader as api

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

def languagemodel(words):
    # English embeddings http://vectors.nlpl.eu/repository/ (English CoNLL17 corpus)
    model = gensim.models.KeyedVectors.load("../../../cds-lang-data/word2vec_models/english/english_word2vec.bin")

    # the list of words we want to plot
    

    # an empty list for vectors
    X = []
    # get vectors for subset of words
    for word in words:
        X.append(model[word])

    # Use PCA for dimensionality reduction to 2D
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    # or try SVD - how are they different?
    svd = TruncatedSVD(n_components=2)
    # fit_transform the initialized PCA model
    result = svd.fit_transform(X)

    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])

    # for each word in the list of words
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.show()

languagemodel(words)