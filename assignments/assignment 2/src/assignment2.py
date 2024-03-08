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

# Saving models
from joblib import dump, load

def vectorizer(): # creates a vectorizer to use in the classifiers
    # creating a vectorizer object with the following parameters:
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams
                                lowercase =  True,       # make everything lowercase
                                max_df = 0.95,           # remove 5% most common words
                                min_df = 0.05,           # remove 5% rarest words
                                max_features = 500)      # keep top 500 features
    # saving the vectorizer for use in the models
    model_folder = os.path.join("..", "models")
    dump(vectorizer, model_folder + "/" + "tfidf_vectorizer.joblib")

def logisticregression(): # creates the logistic regression classifier
    # load the data to a pandas csv
    filename = os.path.join("..", "..", "..", "..", "cds-lang-repo", "cds-language", "data", "fake_or_real_news.csv")
    data = pd.read_csv(filename, index_col=0)

    # create data variables containing data and labels
    X = data["text"]
    y = data["label"]

    # creating a 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,           # texts for the model
                                                        y,          # classification labels
                                                        test_size=0.2,   # 80/20 split
                                                        random_state=42) # random state for reproducibility

    # loading the saved vectorized data from the models folder
    model_folder = os.path.join("..", "models")
    vectorizer = load(model_folder + "/" + "tfidf_vectorizer.joblib")

    # fit the vectorizer to the data
    X_train_feats = vectorizer.fit_transform(X_train) # fit to the training data
    X_test_feats = vectorizer.transform(X_test) # fit for the test data
    feature_names = vectorizer.get_feature_names_out() # get feature names

    # fit the classifier to the data
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)

    # get predictions
    y_pred = classifier.predict(X_test_feats)

    # show 20 most informative features
    clf.show_features(vectorizer, y_train, classifier, n=20)

    # use confusion matrix to check performance
    metrics.ConfusionMatrixDisplay.from_estimator(classifier,           # the classifier name
                                                X_train_feats,          # the training features
                                                y_train,                # the training labels
                                                cmap=plt.cm.Blues,      # make the colours prettier
                                                labels=["FAKE", "REAL"])# the labels in the data arranged alphabetically

    # get classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    print(classifier_metrics)

    # saving classification report as a .txt file
    text_file = open(r'../output/LR_classification_report.txt', 'w')
    text_file.write(classifier_metrics)
    text_file.close()

    # cross validation
    X_vect = vectorizer.fit_transform(X)

    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = LogisticRegression(random_state=42)
    clf.plot_learning_curve(estimator, title, X_vect, y, cv=cv, n_jobs=4)

    # saving the model for later use
    model_folder = os.path.join("..", "models")
    dump(classifier, model_folder + "/" + "LR_classifier.joblib")

    # loading the saved model to the models folder
    model_folder = os.path.join("..", "models")
    loaded_clf = load(model_folder + "/" + "LR_classifier.joblib")

    # test sentence
    sentence = "Hillary Clinton is a crook who eats babies!"

    # prediction for the test sentence
    test_sentence = vectorizer.transform([sentence])
    print(sentence)
    print(loaded_clf.predict(test_sentence))

def neuralnetwork(): # creates the neural network classifier
    # load the data to a pandas csv
    filename = os.path.join("..", "..", "..", "..", "cds-lang-repo", "cds-language", "data", "fake_or_real_news.csv")
    data = pd.read_csv(filename, index_col=0)

    # create data variables containing data and labels
    X = data["text"]
    y = data["label"]

    # creating a 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,           # texts for the model
                                                        y,          # classification labels
                                                        test_size=0.2,   # 80/20 split
                                                        random_state=42) # random state for reproducibility

    # loading the saved vectorized data from the models folder
    model_folder = os.path.join("..", "models")
    vectorizer = load(model_folder + "/" + "tfidf_vectorizer.joblib")

    # fit the vectorizer to the data
    X_train_feats = vectorizer.fit_transform(X_train) # fit to the training data
    X_test_feats = vectorizer.transform(X_test) # fit for the test data
    feature_names = vectorizer.get_feature_names_out() # get feature names

    #create a classifier
    classifier = MLPClassifier(activation = "logistic",
                            hidden_layer_sizes = (20,), # 20 neurons in our hidden layer
                            max_iter=1000, # 1000 epochs
                            random_state = 42) # random state for reproducibility

    # fit the classifier to the data
    classifier.fit(X_train_feats, y_train)

    # get predictions
    y_pred = classifier.predict(X_test_feats)

    # use confusion matrix to check performance
    metrics.ConfusionMatrixDisplay.from_estimator(classifier,           # the classifier name
                                                X_train_feats,          # the training features
                                                y_train,                # the training labels
                                                cmap=plt.cm.Blues,      # make the colours prettier
                                                labels=["FAKE", "REAL"])# the labels in the data arranged alphabetically

    # get classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    print(classifier_metrics)

    # saving classification report as a .txt file
    text_file = open(r'../output/MLP_classification_report.txt', 'w')
    text_file.write(classifier_metrics)
    text_file.close()

    # plotting loss curves. should ideally have a smooth, steep downwards slope which ends in a plateau
    plt.plot(classifier.loss_curve_)
    plt.title("Loss curve during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')
    plt.show()

    # saving the model for later use
    model_folder = os.path.join("..", "models")
    dump(classifier, model_folder + "/" + "MLP_classifier.joblib")

    # loading the saved model to the models folder
    model_folder = os.path.join("..", "models")
    loaded_clf = load(model_folder + "/" + "MLP_classifier.joblib")

    # test sentence
    sentence = "Hillary Clinton is a crook who eats babies!"

    # prediction for the test sentence
    test_sentence = vectorizer.transform([sentence])
    print(sentence)
    print(loaded_clf.predict(test_sentence))

if __name__ =="__main__":
    vectorizer()
    logisticregression()
    neuralnetwork()