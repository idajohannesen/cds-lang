from codecarbon import EmissionsTracker
import os
# CodeCarbon tracker
outfolder = os.path.join("..", "assignment 5", "input") # set output folder
tracker = EmissionsTracker(project_name="assignment 4", # create tracker
                           experiment_id="emotion analysis of GoT",
                           output_dir=outfolder,
                           output_file="emissions.csv")
tracker.start_task("load packages")

# creating dataframes
import pandas as pd

# loading model
from transformers import pipeline

# visualization
import matplotlib.pyplot as plt

tracker.stop_task()

def load_classifier():
    """
    Loads classifier pipeline from HuggingFace
    
    Returns:
        classifier: emotion classifier
    """
    classifier = pipeline("text-classification", 
                          model="j-hartmann/emotion-english-distilroberta-base", 
                          return_all_scores=False) # return only the most likely score
    return classifier

def load_data():
    """
    Loads the data
    
    Returns:
        data: dataset as pandas dataframe
    """
    filename = os.path.join("input/Game_of_Thrones_Script.csv") # navigating into the input folder
    data = pd.read_csv(filename, keep_default_na=False) # reading the dataset as a pandas dataframe
    return data

def get_scores(data, classifier):
    """
    Adds an emotion label to every line in the dataset

    Arguments:
        data: dataset as pandas dataframe
        classifier: emotion classifier
    Returns:
        scores: list of emotion labels + likeliness scores
    """
    i = 1
    scores = [] # make an empty list for every emotion label to go into
    for row in data["Sentence"]:
        result = classifier(row)
        scores.append(result)
        print("finished row " + str(i))
        i += 1
    return scores

def get_labels(scores):
    """
    Removes the scores from the list 'scores' and leaving only the label names
    
    Arguments:
        scores: list of emotion labels + likeliness scores
    Returns:
        emotions: list of emotion labels
    """
    row = 0 # create a row counter to use for the slice
    emotions = [] # create list for emotion labels to go in

    for emotion in scores:
        emotion = scores[row][0]["label"] #remove everything but the emotion label using slice
        emotions.append(emotion) # add to list
        row += 1 # go to next row
    return emotions

def update_dataframe(emotions, data):
    """
    Adds the emotion labels as an extra column in the dataframe
    
    Arguments:
        emotions: list of emotion labels
        data: dataset as pandas dataframe
    Returns:
        new_df: updated dataframe with emotion labels
    """
    # Using DataFrame.insert() to add a column
    new_df = data # clone the dataframe before adding to it
    new_df.insert(6, #insert as the 6th column
                "Emotion_label", #column name
                emotions, # data to insert
                True) # allow duplicates
    new_df.to_csv("output/dataframe.csv", index=False) # upload dataframe to output folder
    return new_df

def emotion_distribution_plot(new_df):
    """
    Creates a plot for the distribution of emotion labels for every season
    
    Arguments:
        new_df: updated dataframe with emotion labels
    """
    season_list = ["Season 1", "Season 2", "Season 3", "Season 4", "Season 5", "Season 6", "Season 7", "Season 8"]
    for season in season_list:
        season_label = season # create a label for every season
        season_df = new_df[new_df['Season'] == season_label] #split dataframe based on seasons
        sorted_df = season_df.sort_values('Emotion_label', inplace=False) # sort the dataframe so the plots appear consistently
        
        plt.figure(figsize = (16,6))
        plt.hist(sorted_df["Emotion_label"], bins = 7) # plot histogram
        plt.title(season_label)
        plt.xlabel('Emotion labels')
        plt.ylabel('Frequencies')

        plt.savefig('output/' + season_label + '.png') # save output
        plt.show()
        plt.clf()
    
def season_distribution_plot(new_df):
    """
    Creates a plot for the relative frequency of every emotion label across seasons

    Arguments:
        new_df: updated dataframe with emotion labels
    """  
    emotion_list = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    season_list = ["Season 1", "Season 2", "Season 3", "Season 4", "Season 5", "Season 6", "Season 7", "Season 8"]
    for emotions in emotion_list:
        rel_freq_list = [] # save relative frequencies per emotion
        for seasons in season_list:
            emotion_df = new_df[(new_df["Emotion_label"] == emotions)&(new_df["Season"] == seasons)] #split dataframe based on seasons
            rel_freq = len(emotion_df)/len(new_df[new_df["Season"]==seasons])*100 # calculate relative frequency
            rel_freq_list.append(rel_freq)
        
        plt.figure(figsize = (16,6))
        plt.plot(season_list, rel_freq_list) # plot
        plt.title(emotions)
        plt.xlabel('Seasons')
        plt.ylabel('Relative frequencies')

        plt.savefig('output/' + emotions + '.png') # save output
        plt.show()
        plt.clf()

def main():
    tracker.start_task("load model")
    classifier = load_classifier()
    tracker.stop_task()

    tracker.start_task("load data")
    data = load_data()
    tracker.stop_task()

    tracker.start_task("sentiment analysis")
    scores = get_scores(data, classifier)
    tracker.stop_task()

    tracker.start_task("fix labels")
    emotions = get_labels(scores)
    tracker.stop_task()

    tracker.start_task("update dataframe")
    new_df = update_dataframe(emotions, data)
    tracker.stop_task()

    tracker.start_task("plot distribution of emotion labels for every season")
    emotion_distribution_plot(new_df)
    tracker.stop_task()

    tracker.start_task("plot emotion freq across seasons")
    season_distribution_plot(new_df)
    tracker.stop_task()
    
    tracker.stop()

if __name__=="__main__":
    main()