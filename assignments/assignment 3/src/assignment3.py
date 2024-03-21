import os
import argparse
import gensim.downloader as api
import pandas as pd

def parser(): # create arguments for the artist and input word
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

def load_data(): # load the spotify song dataset
    filename = os.path.join("../input/Spotify_Million_Song_Dataset_exported.csv")
    data = pd.read_csv(filename)
    return data

def download_model(): # download a pretrained model
    model = api.load("glove-wiki-gigaword-50")
    return model

def find_similar_words(model, args): # finding top 10 most similar words to the input word
    most_similar=model.most_similar(args.input, topn=10)
    words, similarity = zip(*most_similar) # split the output into a word variable and a similarity variable
    return words
    print(words)

def select_artist(data, args): # create a list of the chosen artist's lyrics
    search_docs = list(data[data["artist"]==args.artist]["text"])
    return search_docs

def counter(search_docs, words): # count how many times the words appear in the lyrics
    doc_counter = 0
    for doc in search_docs:
        if any(word in doc.split() for word in words):
            doc_counter += 1
    return doc_counter

def print_results(doc_counter, search_docs, args):
    percent = round(doc_counter/len(search_docs)*100, 1) # calculate percentages and round to a single decimal point
    print(percent, "% of", args.artist, "'s songs contain words related to", args.input) # print the results

def main():
    args = parser()
    data = load_data()
    model = download_model()
    words = find_similar_words(model, args)
    search_docs = select_artist(data, args)
    doc_counter = counter(search_docs, words)
    print_results(doc_counter, search_docs, args)

if __name__=="__main__":
    main()