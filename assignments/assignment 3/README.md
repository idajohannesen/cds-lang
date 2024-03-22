Short description:
This code calculates how many songs from a given artist include words related to an input word. It takes both the artist and input word as arguments which can be decided by the user when running the code. Results are added to a .txt file in the output folder.

Data:
The dataset is a corpus of lyrics from 57,650 English-language songs. It can be inserted as a csv file in the input folder. 
The data can be downloaded from here: https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs

Reproducing:
A setup file has been included which can be run with 'source' to create a virtual environment with the necessary requirements. The script requires that a csv file of the data is located in the input folder. The code can be run through the command line by inputting "python assignment3.py -i [word input] -a [artist]" from the src folder. The artist must be part of the dataset.

Discussion/summary:
While the model runs successfully, the results may be skewed due to issues with the dataset.
For example, the dataset is not properly cleaned. One can find lyrics which contain non-lyrics such as "(verse 1)" or lyrics in other languages than English, which the model won't work for.
Additionally, the dataset does not always contain the full discography of every artist, and the results may therefore not represent the actual discography for the artist. The sample size of the included discography can also affect the accuracy.

The list of most similar words also tend to include antonyms, which is likely not what the user wants to count when selecting their word. The quality of the list of similar words can therefore affect the result.