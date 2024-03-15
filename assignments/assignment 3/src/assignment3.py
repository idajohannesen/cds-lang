# run pipreqs src --savepath requirements.txt to find all requirements

import os
import argparse
import gensim.downloader as api

# load the song lyric data
# downloads a word embedding model
model = api.load("glove-wiki-gigaword-50")
# input a word and find similar words through word embeddings
# finding how many songs for a given artist feature these terms
# print results

if __name__=="__main__":
    main()