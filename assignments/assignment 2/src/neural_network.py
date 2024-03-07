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

def neuralnetworkmodel():
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
    from joblib import dump, load
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
    from joblib import dump, load
    model_folder = os.path.join("..", "models")
    dump(classifier, model_folder + "/" + "MLP_classifier.joblib")

    # loading the saved model to the models folder
    from joblib import dump, load
    model_folder = os.path.join("..", "models")
    loaded_clf = load(model_folder + "/" + "MLP_classifier.joblib")

    # test sentence
    sentence = "Hillary Clinton is a crook who eats babies!"

    # prediction for the test sentence
    test_sentence = vectorizer.transform([sentence])
    print(sentence)
    print(loaded_clf.predict(test_sentence))