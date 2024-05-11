from codecarbon import EmissionsTracker
import os
# CodeCarbon tracker
outfolder = os.path.join("..", "assignment 5", "input") # set output folder
tracker = EmissionsTracker(project_name="assignment 3", # create tracker
                           experiment_id="song lyrics related to input word",
                           output_dir=outfolder,
                           output_file="emissions.csv")
tracker.start_task("load packages")

# creating arguments
import argparse

# loading model
import gensim.downloader as api

# creating dataframes
import pandas as pd

tracker.stop_task()

def parser():
    """
    Creates arguments for the artist and input word
    
    Returns:
        args: custom set arguments for input word and artist
    """
    parser = argparse.ArgumentParser(description="Choosing an artist and a word")
    parser.add_argument("--artist",
                        "-a",
                        required=True,
                        help="Name of artist from the database")
    parser.add_argument("--input",
                        "-i",
                        required=True,
                        help="Word to search for")
    args = parser.parse_args()
    return args

def load_data():
    """
    Loads the spotify song dataset
    
    Returns:
        data: dataset as pandas dataframe
    """
    filename = os.path.join("input", "Spotify Million Song Dataset_exported.csv")
    data = pd.read_csv(filename)
    data["artist"] = data["artist"].str.lower()
    return data

def download_model():
    """
    Downloads a pretrained model

    Returns:
        model: language model
    """
    model = api.load("glove-wiki-gigaword-50")
    return model

def find_similar_words(model, args):
    """
    Finds the top 10 most similar words to the input word
    
    Arguments:
        model: language model
        args: custom set arguments for input word and artist
    Returns:
        words: list of words being searched for, based on input word
    """
    most_similar=model.most_similar(args.input, topn=10)
    words, similarity = zip(*most_similar) # split the output into a word variable and a similarity variable
    # adding the input word to the list of words being searched for
    words = list(words)
    words.append(args.input)
    words = tuple(words)
    return words

def select_artist(data, args):
    """
    Creates a list of the chosen artist's lyrics

    Arguments:
        data: dataset as pandas dataframe
        args: custom set arguments for input word and artist
    Returns:
        search_docs: list of all lyrics for the chosen artist
    """
    search_artist = args.artist.lower()
    search_docs = list(data[data["artist"]==search_artist]["text"])
    return search_docs

def counter(search_docs, words):
    """
    Counts how many times the input word and similar words appear in the lyrics

    Arguments:
        search_docs: list of all lyrics for the chosen artist
        words: list of words being searched for, based on input word
    Returns:
        doc_counter: number of input word appearances
    """
    doc_counter = 0
    for doc in search_docs:
        if any(word in doc.split() for word in words):
            doc_counter += 1
    return doc_counter

def print_results(doc_counter, search_docs, args):
    """
    Prints the results

    Arguments:
        doc_counter: number of input word appearances
        search_docs: list of all lyrics for the chosen artist
        args: custom set arguments for input word and artist
    """
    percent = round(doc_counter/len(search_docs)*100, 1) # calculate percentages and round to a single decimal point
    results = "{}% of {}'s songs contain words related to {}".format(percent, args.artist, args.input) # print the results
    print(percent, "% of", args.artist, "'s songs contain words related to", args.input) # print the results
    # saving the results as a .txt file
    text_file = open(r'output/results.txt', 'a')
    text_file.write(results + "\n")
    text_file.close()

def main():
    tracker.start_task("create arguments")    
    args = parser()
    tracker.stop_task()

    tracker.start_task("load data") 
    data = load_data()
    tracker.stop_task()

    tracker.start_task("load model") 
    model = download_model()
    tracker.stop_task()

    tracker.start_task("find similar words") 
    words = find_similar_words(model, args)
    tracker.stop_task()

    tracker.start_task("select artist") 
    search_docs = select_artist(data, args)
    tracker.stop_task()

    tracker.start_task("counting words")
    doc_counter = counter(search_docs, words)
    tracker.stop_task()

    tracker.start_task("print results") 
    print_results(doc_counter, search_docs, args)
    tracker.stop_task()

    tracker.stop()

if __name__=="__main__":
    main()