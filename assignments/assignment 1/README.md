# Assignment 1

# Short description:
This code loops through the texts of a corpus and extracts linguistic features from them. It extracts the relative frequency of nouns, verbs, adjectives, and adverbs per 10,000 words and the number of unique locations, persons, and organisations mentioned. These features have been gathered in a pandas dataframe for every subfolder and can be found in the ```output``` folder as csv files. Here, the average frequency of the linguistic features can also be found for each subfolder as both a csv files of the exact numbers and a graph offering easy visualization and comparison. Both of these are in the subfolder titled ```average```.

This code uses CodeCarbon to track CO₂ emissions. The results are discussed in Assignment 5.

# Data:
The dataset is the Uppsala Student English corpus, which consists of 1,489 English essays written by Swedish university students. 
The data can be downloaded from here: https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457
Download the file called ```USEcorpus.zip``` and unzip the file. Inside the ```USEcorpus``` folder, there will be a second folder also named ```USEcorpus```, containing the subfolders ```a1```, ```a2```, etc. Place this innermost ```USEcorpus``` folder in the ```input``` folder.

# Reproducing:
A setup file has been included which creates a virtual environment with the necessary requirements, as well as a ```run.sh``` file which activates the virtual environment and runs the script. The script requires that the data is located in the ```input``` folder with the structure mentioned above. 

Start by running the setup file by inputting ```bash setup.sh``` in the terminal. 
The code can then be run through the terminal by inputting ```bash run.sh```.
Both of these lines should be executed from the ```assignment 1``` folder.

The code has been written and tested on a Windows laptop and may not work correctly on other operating systems.

# Discussion/summary:
The code is able to run succesfully and extract features from every text. However, the results should be taken with a grain of salt, as some words may have been tokenized incorrectly which leads to errors when extracting linguistic features.
When looking at the average scores for every subfolder, which can be seen in the ```plot.png``` and ```averages.csv``` files, the numbers for named entity recognition are generally very low. Apart from proper nouns being less common in text that the other word classes, the low number could indicate that the model struggles to identify these entities. None of the three types of named entities stand out from the other two and there is no noticable rising or falling trend of their usage. This may be because of the low numbers leading to very small differences between document subfolders. 
The relative frequencies of word classes are consistently in the same order. There are always more nouns than verbs, more verbs than adjectives, and more adjectives than adverbs. This shows that certain word classes are simply more common than others. The most common word classes, like nouns and verbs, also shown bigger variation throughout, whereas the relative frequency of adjectives and adverbs show a steadier decline with smaller peaks and valleys throughout.
There appears to be a slight trend of lesser frequencies in the later documents, but it is not a steady decline throughout. Having a larger dataset with more document subfolders might have made this trend easier to spot in the visualization.