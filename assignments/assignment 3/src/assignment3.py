import os
import argparse
import gensim.downloader as api
import pandas as pd

def parser():
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
    filename = os.path.join("../input/Spotify_Million_Song_Dataset_exported.csv")
    data = pd.read_csv(filename)
    return data

def download_model():
    model = api.load("glove-wiki-gigaword-50")
    return model

def find_similar_words(model, args):
    most_similar=model.most_similar(args.input, topn=10)
    words, similarity = zip(*most_similar)
    return words
    print(words)

def select_artist(data, args):
    search_docs = list(data[data["artist"]==args.artist]["text"])
    return search_docs

def counter(search_docs, words):
    doc_counter = 0
    for doc in search_docs:
        if any(word in doc.split() for word in words):
            doc_counter += 1
    return doc_counter

def print_results(doc_counter, search_docs, args):
    percent = round(doc_counter/len(search_docs)*100, 1)
    print(percent, "% of", args.artist, "'s songs contain words related to", args.input)

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