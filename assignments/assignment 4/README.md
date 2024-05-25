# Assignment 4

# Short description: 
This code goes through the script of all 8 seasons of Game of Thrones and assigns each line an emotion label: anger, disgust, fear, joy, neutral, sadness, or surprise. Two plots are created: the distribution of emotions for every season and the relative frequency of an emotion label across all 8 seasons.

This code uses CodeCarbon to track CO₂ emissions. The results are discussed in Assignment 5.

# Data:
The dataset consists of the script for all 8 seasons of the TV show Game of Thrones, split into lines.
The data can be downloaded from here: https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons?select=Game_of_Thrones_Script.csv
Download the data from the link and unzip it. Inside the unzipped folder, there is a csv file titled ```Game_of_Thrones_Script.csv```. Place this csv file in the ```input``` folder.

# Reproducing:
A setup file has been included which creates a virtual environment with the necessary requirements, as well as a ```run.sh``` file which activates the virtual environment and runs the script. The script requires that the data is located in the ```input``` folder with the structure mentioned above. 

Start by running the setup file by inputting ```bash setup.sh``` in the terminal. 
The code can then be run through the terminal by inputting ```bash run.sh```.
Both of these lines should be executed from the ```assignment 4``` folder.

The code has been written and tested on a Windows laptop and may not work correctly on other operating systems.

# Discussion/summary:
The script succesfully creates plots over the distribution of emotion labels and seasons. The vast majority of lines are labeled as being 'neutral', with relative frequencies ranging from around 45% to 52%. By looking at the results in the .csv file, we can see that many of the shorter lines are considered neutral in tone. This makes sense, since many of them are simple replies, yes/no's or names being mentioned. These would not have obvious emotional associations based on the text, but a viewer would have access to the general tone of the line and be able to infer emotion from this. Therefore, the labeling does have its limitations since it is purely text-based and ends up with excessive neutral labels that may not represent the emotions of the overall scenes. 
We can note that disgust is generally decreasing. Our plots also show that anger, surprise, and disgust are consistently the most common emotions after the neutral label. Joy is often the least frequent label in a season, except for season 8 and 5 where fear is the least frequent emotion. Based on this, Game of Thrones comes off as a series full of conflict and only a few moments of joy per season. 