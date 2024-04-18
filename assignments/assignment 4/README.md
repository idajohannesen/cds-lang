Short description: This code goes through the script of all 8 seasons of Game of Thrones and assigns each line an emotion label: anger, disgust, fear, joy, neutral, sadness, or surprise. Two plots are created: the distribution of emotions for every season and the relative frequency of an emotion label across all 8 seasons.

Data:
The dataset consists of scripts from all seasons of the TV show Game of Thrones, split into lines. It can be inserted as a .csv file in the input folder.
The data can be downloaded from here: https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons?select=Game_of_Thrones_Script.csv

Reproducing:
A setup file has been included which can be run to create a virtual environment with the necessary requirements. The script requires that a csv file of the data is located in the input folder. The code can be run through the command line by inputting "python assignment4.py" from the src folder.

Discussion/summary:
The script succesfully creates plots over the distribution of emotion labels and seasons. The vast majority of lines are labeled as being 'neutral', with relative frequencies ranging from around 45% to 52%. By looking at the results in the .csv file, we can see that many of the shorter lines are considered neutral in tone. This makes sense, since many of them are simple replies, yes/no's or names being mentioned. These would not have obvious emotional associations based on the text, but a viewer would have access to the general tone of the line and be able to infer emotion from this. Therefore, the labeling does have its limitations since it is purely text-based and ends up with excessive neutral labels. 
We can note that disgust is generally decreasing. Our plots also show that anger, surprise, and disgust are the most common emotions after the neutral label. Joy is often the least frequent label in a season, except for season 8 and 5 where fear is the least frequent emotion. Based on this, Game of Thrones comes off as a series full of conflict and only a few moments of joy per season. 