import os
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

def load_classifier(): # load classifier pipeline from HuggingFace
    classifier = pipeline("text-classification", 
                          model="j-hartmann/emotion-english-distilroberta-base", 
                          return_all_scores=False) # return only the most likely score
    return classifier

def load_data(): # navigating into the input folder and reading the dataset as a pandas dataframe
    filename = os.path.join("../input/Game_of_Thrones_Script.csv")
    data = pd.read_csv(filename, keep_default_na=False)
    return data

def get_scores(data, classifier): # using a for loop to get an emotion label for every line in the dataset
    i = 1
    scores = [] # make an empty list for every emotion label to go into
    for row in data["Sentence"]:
        result = classifier(row)
        scores.append(result)
        print("finished row " + str(i))
        i += 1
    return scores

def get_labels(scores): # removing the scores from the list 'scores' and leaving only the label names
    row = 0 # create a row counter to use for the slice
    emotions = [] # create list for emotion labels to go in

    for emotion in scores:
        emotion = scores[row][0]["label"] #remove everything but the emotion label using slice
        emotions.append(emotion) # add to list
        row += 1 # go to next row
    return emotions

def update_dataframe(emotions, data): # add the emotion labels as an extra column in the dataframe
    # Using DataFrame.insert() to add a column
    new_df = data # clone the dataframe before adding to it
    new_df.insert(6, #insert as the 6th column
                "Emotion_label", #column name
                emotions, # data to insert
                True) # allow duplicates
    new_df.to_csv("../output/dataframe.csv", index=False) # upload dataframe to output folder
    return new_df

def emotion_distribution_plot(new_df): # create a plot for the distribution of emotion labels for every season
    counter = 1 # create a counter to update the season labels
    season_list = ["Season 1", "Season 2", "Season 3", "Season 4", "Season 5", "Season 6", "Season 7", "Season 8"]
    season_length = []
    for season in season_list:
        season_label = season # create a label for every season
        season_df = new_df[new_df['Season'] == season_label] #split dataframe based on seasons
        sorted_df = season_df.sort_values('Emotion_label', inplace=False) # sort the dataframe so the plots appear consistently
        length = len(sorted_df)
        season_length.append(length) # save the total amount of lines for each season, is used later to plot relative frequencies
        
        plt.figure(figsize = (16,6))
        plt.hist(sorted_df["Emotion_label"], bins = 7) # plot histogram
        plt.title(season_label)
        plt.xlabel('Emotion labels')
        plt.ylabel('Frequencies')

        plt.savefig('../output/' + season_label + '.png') # save output
        plt.show()
        plt.clf()
        
        counter += 1
        return season_list, season_length
    
def season_distribution_plot(new_df, season_length): # create a plot for the relative frequency of every emotion label across seasons    
    emotion_list = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    emotion_counter = 0
    season_list = ["Season 1", "Season 2", "Season 3", "Season 4", "Season 5", "Season 6", "Season 7", "Season 8"]
    season_counter = 1
    for emotions in emotion_list:
        rel_freq_list = [] # save relative frequencies per emotion
        length_counter = 0 # create counter to loop through the list of every season's length
        for seasons in season_list:
            emotion_df = new_df[(new_df["Emotion_label"] == emotions)&(new_df["Season"] == seasons)] #split dataframe based on seasons
            #emotion_amt = float(len(emotion_df))
            #total_amt = float(season_length[length_counter])
            rel_freq = len(emotion_df)/len(new_df[new_df["Season"]==seasons])*100 # calculate relative frequency
            rel_freq_list.append(rel_freq)
            length_counter += 1
        
        plt.figure(figsize = (16,6))
        plt.plot(season_list, rel_freq_list) # plot
        plt.title(emotions)
        plt.xlabel('Seasons')
        plt.ylabel('Relative frequencies')

        plt.savefig('../output/' + emotions + '.png') # save output
        plt.show()
        plt.clf()
        
        emotion_counter += 1
        season_counter += 1

def main():
    classifier = load_classifier()
    data = load_data()
    scores = get_scores(data, classifier)
    emotions = get_labels(scores)
    new_df = update_dataframe(emotions, data)
    season_length = emotion_distribution_plot(new_df)
    season_distribution_plot(new_df, season_length)

if __name__=="__main__":
    main()