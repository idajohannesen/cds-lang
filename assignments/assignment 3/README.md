# Assignment 3

# Short description:
This code calculates how many songs from a given artist include words related to an input word. The result is presented as a percentage. It takes both the artist and input word as arguments which can be decided by the user when running the code. Results are added to a .txt file in the ```output``` folder.

This code uses CodeCarbon to track CO₂ emissions. The results are discussed in Assignment 5.

# Data:
The dataset is a corpus of lyrics from 57,650 English-language songs. It can be inserted as a .csv file in the ```input``` folder. 
The data can be downloaded from here: https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs

# Reproducing:
A setup file has been included which creates a virtual environment with the necessary requirements, as well as a ```run.sh``` file which activates the virtual environment and runs the script. The script requires that the data is located in the ```input``` folder with the structure mentioned above. 

Start by running the setup file by inputting ```bash setup.sh``` in the terminal. 
The code can then be run through the terminal by inputting ```bash run.sh```. By default, the code searches for the word ```sadness``` and the artist ```Adele```, but this can be changed in the ```run.sh``` script. The artist must be part of the dataset. Capitalizing any letter of the input word and artist name does not affect the final results.
Both of these lines should be executed from the ```assignment 3``` folder.

The code has been written and tested on a Windows laptop and may not work correctly on other operating systems.

# Discussion/summary:
The model has been run with four examples: ABBA and joy, Cher and love, and Adele with both sadness and love. The results, which can also be found in the ```results.txt``` file in the ```output``` folder, were:
42.5% of ABBA's songs contain words related to joy
96.3% of Cher's songs contain words related to love
5.6% of Adele's songs contain words related to sadness
100.0% of Adele's songs contain words related to love

From these results, although they are very limited and nothing conclusive can be determined without many more runs, we can see that love and joy is present in all three artists' songs. Only 5.6% of Adele's songs featured sadness as part of the lyrics, while all of her songs appear to be about love in some capacity. This could suggest that emotions like joy or being in love are common topics in music by these three artists. 
The script can be used to perform analyses like the one above, for example by comparing an input word across many different artists, or various different input words for a single artist's songs.

While the model runs successfully, the results may be skewed due to issues with the dataset.
For example, the dataset is not properly cleaned. One can find lyrics which contain non-lyrics such as "(verse 1)" or lyrics in other languages than English, which the model will not work for.
Additionally, the dataset does not always contain the full discography of every artist, and the lyrics in the dataset may therefore not represent the full discography for the artist. The limited sample size of the included discography can also affect the accuracy of the calculated score as many of the artists only have a few songs included in the dataset. Smaller datasets tend to be less accurate in the end as only a few errors will massively skew the final result. This is especially noticable with the example run which outputted that "100.0% of Adele's songs contain words related to love." Adele only has 12 songs in the dataset, which are not likely to be entirely accurate of her entire discography. This result does not guarantee that every single Adele song is about love. The accuracy is generally better for artists with more songs present in the dataset.

The list of most similar words also tend to include antonyms, which is likely not what the user wants to count when selecting their word. The quality of the list of similar words can therefore affect the final output and give a misleading result. The results should therefore be taken with a grain of salt and are likely more accurate to interpret as how often the word appears as a broad topic, which both includes synonyms and antonyms for the input word. For example, a word like 'joy' may also include 'sadness' as a similar word, which means that the final result is a count of how many times joy is discussed as well as the absence of it. This may ultimately not be what the user expects from the code but it is important to consider when interpreting results. 