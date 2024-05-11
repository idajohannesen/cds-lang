from codecarbon import EmissionsTracker
import os
# CodeCarbon tracker
outfolder = os.path.join("..", "assignment 5", "input") # set output folder
tracker = EmissionsTracker(project_name="assignment 2 logistic regression", # create tracker
                           experiment_id="logistic regression",
                           output_dir=outfolder,
                           output_file="emissions.csv")
tracker.start_task("load packages")

# system tools
import sys

# data munging tools
import pandas as pd
import classifier_utils as clf

# machine learning
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# visualisation
import matplotlib.pyplot as plt

# saving and loading models
from joblib import dump, load

tracker.stop_task()

def load_data(): # load the data to a pandas csv
    """
    Load the data to a pandas csv

    Returns:
        data: dataset as pandas dataframe
    """
    filename = os.path.join("input", "fake_or_real_news.csv")
    data = pd.read_csv(filename, index_col=0) # reading data csv file
    return data
    
def create_variables(data):
    """
    Creates data variables containing data and labels

    Arguments:
        data: dataset as pandas dataframe
    Returns:
        X: text
        y: real/fake label
    """
    X = data["text"]
    y = data["label"]
    return X, y
    
def split(X, y):
    """
    Creates a 80/20 train-test split

    Arguments:
        X: text
        y: real/fake label
    Returns:
        X_train: texts for training
        X_test: texts for testing
        y_train: labels for training data
        y_test: labels for testing data
    """
    X_train, X_test, y_train, y_test = train_test_split(X,           # texts for the model
                                                        y,          # classification labels
                                                        test_size=0.2,   # 80/20 split
                                                        random_state=42) # random state for reproducibility
    return X_train, X_test, y_train, y_test

def load_vectorizer():
    """
    Loads the saved vectorized data from the models folder

    Returns:
        vectorizer: vectorizer model
    """
    model_folder = os.path.join("models")
    vectorizer = load(model_folder + "/" + "tfidf_vectorizer.joblib")
    return vectorizer
    
def fit_vec_to_data(vectorizer, X_train, X_test):
    """
    Fits the vectorizer to the data

    Arguments:
        vectorizer: vectorizer model
        X_train: texts for training
        X_test: texts for testing
    Returns:
        X_train_feats: texts for training, vectorized
        X_test_feats: texts for testing, vectorized
    """
    X_train_feats = vectorizer.fit_transform(X_train) # fit to the training data
    X_test_feats = vectorizer.transform(X_test) # fit for the test data
    return X_train_feats, X_test_feats
    
def fit_classifier_to_data(X_train_feats, y_train):
    """
    Creates and fits the classifier to the data

    Arguments:
        X_train_feats: texts for training, vectorized
        y_train: labels for training data
    Returns:
        classifier: logistic regression classifier
    """
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train) # includes both the creation and fitting of the classifier
    return classifier

def predict(X_test_feats, vectorizer, y_train, classifier):
    """
    Gets predictions

    Arguments:
        X_train_feats: texts for training, vectorized
        vectorizer: vectorizer model
        y_train: labels for training data
        classifier: logistic regression classifier
    Returns:
        y_pred: predictions
    """
    y_pred = classifier.predict(X_test_feats)

    clf.show_features(vectorizer, y_train, classifier, n=20) # get 20 most informative features
    return y_pred

def confusion_matrix(classifier, X_train_feats, y_train):
    """
    Checks performance with a confusion matrix

    Arguments:
        classifier: logistic regression classifier
        X_train_feats: texts for training, vectorized
        y_train: labels for training data
    """
    metrics.ConfusionMatrixDisplay.from_estimator(classifier,           # the classifier name
                                                X_train_feats,          # the training features
                                                y_train,                # the training labels
                                                cmap=plt.cm.Blues,      # make the colours prettier
                                                labels=["FAKE", "REAL"])# the labels in the data arranged alphabetically
    plt.savefig("output/confusion_matrix_lr.png")
    plt.close()

def classification_report(y_test, y_pred):
    """
    Gets classification report

    Arguments:
        y_test: labels for testing data
        y_pred: predictions
    """
    classifier_metrics = metrics.classification_report(y_test, y_pred) # making classification report
    print("Classification report")
    print(classifier_metrics)
        
    # saving classification report as a .txt file
    text_file = open(r'output/classification_report_lr.txt', 'w')
    text_file.write(classifier_metrics)
    text_file.close()

def cross_validation(vectorizer, X, y):
    """
    Creates cross validation graph
    
    Arguments:
        vectorizer: vectorizer model
        X: text
        y: real/fake label
    """
    X_vect = vectorizer.fit_transform(X) # vectorizes all text

    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = LogisticRegression(random_state=42)
    clf.plot_learning_curve(estimator, title, X_vect, y, cv=cv, n_jobs=4) # plot learning curve
    plt.savefig("output/cross_validation.png") # save visualization

def save_model(classifier):
    """
    Saves the model for later use

    Arguments:
        classifier: logistic regression classifier
    """
    model_folder = os.path.join("models")
    dump(classifier, model_folder + "/" + "LR_classifier.joblib")

def load_model():
    """
    Loads the saved model from the models folder

    Returns:
        loaded_clf: loaded classifier model
    """
    model_folder = os.path.join("models")
    loaded_clf = load(model_folder + "/" + "LR_classifier.joblib")
    return loaded_clf

def test_prediction(vectorizer, loaded_clf):
    """
    Gets a prediction for a test sentence

    Arguments:
        vectorizer: vectorizer model
        loaded_clf: loaded classifier model
    """
    sentence = "Hillary Clinton is a crook who eats babies!" # test sentence

    # prediction for the test sentence
    test_sentence = vectorizer.transform([sentence]) # vectorize test sentence
    prediction = str(loaded_clf.predict(test_sentence)) # get prediction as string    print(sentence)
    print(sentence)
    print(prediction)
    
    # save example prediction
    text_file = open(r'output/prediction_lr.txt', 'w') # save as .txt file
    text_file.write(sentence + '\n')
    text_file.write(prediction)
    text_file.close()

def main():
    tracker.start_task("load data")
    data = load_data()
    tracker.stop_task()

    tracker.start_task("create data variables")
    X, y = create_variables(data)
    tracker.stop_task()
    
    tracker.start_task("creating train-test split")
    X_train, X_test, y_train, y_test = split(X, y)
    tracker.stop_task()

    tracker.start_task("load vectorizer")
    vectorizer = load_vectorizer()
    tracker.stop_task()

    tracker.start_task("fit vectorizer")
    X_train_feats, X_test_feats = fit_vec_to_data(vectorizer, X_train, X_test)
    tracker.stop_task()

    tracker.start_task("fit classifier")
    classifier = fit_classifier_to_data(X_train_feats, y_train)
    tracker.stop_task()
    
    tracker.start_task("make predictions")
    y_pred = predict(X_test_feats, vectorizer, y_train, classifier)
    tracker.stop_task()

    tracker.start_task("create confusion matrix")
    confusion_matrix(classifier, X_train_feats, y_train)
    tracker.stop_task()

    tracker.start_task("make classification report")
    classification_report(y_test, y_pred)
    tracker.stop_task()

    tracker.start_task("create cross validation graph")
    cross_validation(vectorizer, X, y)
    tracker.stop_task()

    tracker.start_task("save logistic regression model")
    save_model(classifier)
    tracker.stop_task()

    tracker.start_task("load model")
    loaded_clf = load_model()
    tracker.stop_task()

    tracker.start_task("get predictions for test sentence")
    test_prediction(vectorizer, loaded_clf)    
    tracker.stop_task()

    tracker.stop()

if __name__ =="__main__":
    main()