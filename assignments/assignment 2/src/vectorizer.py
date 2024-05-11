from codecarbon import EmissionsTracker
import os
# CodeCarbon tracker
outfolder = os.path.join("..", "assignment 5", "input") # set output folder
tracker = EmissionsTracker(project_name="assignment 2 vectorizer", # create tracker
                           experiment_id="vectorizer",
                           output_dir=outfolder,
                           output_file="emissions.csv")
tracker.start_task("load packages")

# scikit-learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# saving and loading models
from joblib import dump, load
tracker.stop_task()

def vectorizer():
    """
    Creates a vectorizer to use in the classifier models
    """
    # creating a vectorizer object with the following parameters:
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams
                                lowercase =  True,       # make everything lowercase
                                max_df = 0.95,           # remove 5% most common words
                                min_df = 0.05,           # remove 5% rarest words
                                max_features = 500)      # keep top 500 features
    # saving the vectorizer
    model_folder = os.path.join("models")
    dump(vectorizer, model_folder + "/" + "tfidf_vectorizer.joblib")

def main():
    tracker.start_task("create vectorizer")
    vectorizer()
    tracker.stop_task()
    tracker.stop()

if __name__ =="__main__":   
    main()