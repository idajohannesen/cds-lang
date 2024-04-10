Short description:
This code loops through the texts of a corpus and extracts inguistic features from them. It extracts the relative frequency of nouns, verbs, adjectives, and adverbs per 10,000 words and the number of unique locations, persons, and organisations mentioned. These features have been gathered in a pandas dataframe for every subfolder and can be found in the output folder as .csv files.

Data:
The dataset is the Uppsala Student English corpus, which consists of 1,489 English essays written by Swedish university students. It should be inserted in the input folder as a folder named USEcorpus, with the text files found in subfolders a1, a2, etc.
The data can be downloaded from here: insert link

Reproducing:
A setup file has been included which can be run with 'source' to create a virtual environment with the necessary requirements. The script requires that the data is located in the input folder with the structure mentioned above. The code can be run through the command line by inputting "python assignment1.py" from the src folder.

Discussion/summary:
The code is able to run succesfully and extract features from every text. However, the results should be taken with a grain of salt, as some words may have been tokenized incorrectly which leads to errors when extracting linguistic features.