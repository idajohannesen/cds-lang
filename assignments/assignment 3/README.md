# Assignment 3

# Short description:
This code calculates how many songs from a given artist include words related to an input word. It takes both the artist and input word as arguments which can be decided by the user when running the code. Results are added to a .txt file in the ```output``` folder.

This code uses CodeCarbon to track CO₂ emissions. The results are discussed in Assignment 5.

# Data:
The dataset is a corpus of lyrics from 57,650 English-language songs. It can be inserted as a .csv file in the ```input``` folder. 
The data can be downloaded from here: https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs

# Reproducing:
A setup file has been included which creates a virtual environment with the necessary requirements, as well as a ```run.sh``` file which activates the virtual environment and runs the script. The script requires that the data is located in the ```input``` folder with the structure mentioned above. 

Start by running the setup file by inputting ```bash setup.sh``` in the terminal. 
The code can then be run through the terminal by inputting ```bash run.sh```. Make sure to replace the two brackets with an input word and an artist in this file before running. The artist must be part of the dataset.
Both of these lines should be executed from the ```assignment 3``` folder.

The code has been written and tested on a Windows laptop and may not work correctly on other operating systems.

# Discussion/summary:
While the model runs successfully, the results may be skewed due to issues with the dataset.
For example, the dataset is not properly cleaned. One can find lyrics which contain non-lyrics such as "(verse 1)" or lyrics in other languages than English, which the model won't work for.
Additionally, the dataset does not always contain the full discography of every artist, and the results may therefore not represent the actual discography for the artist. The sample size of the included discography can also affect the accuracy.

The list of most similar words also tend to include antonyms, which is likely not what the user wants to count when selecting their word. The quality of the list of similar words can therefore affect the result.