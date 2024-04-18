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
    scores = [] # make an empty list for every emotion label to go into
    for row in data["Sentence"]:
        result = classifier(row)
        scores.append(result)
    return scores

def get_labels(scores): # turn the list of scores into a list with only the emotion label
    counter = 0 # create a counter to use for the slice in order to iterate over every row
    emotions = []

    for emotion in scores:
        emotion = scores[counter][0]["label"] # select only the label
        emotions.append(emotion)
        counter += 1
    return emotions

def update_dataframe(emotions): # add the emotion labels as an extra column in the dataframe
    new_df = data # clone the dataframe before adding to it
    new_df.insert(6, #insert as the 6th column
                  "Emotion_label", #column name
                  emotions, # data to insert
                  True) # allow duplicates
    
    # upload dataframe to output folder
    outpath = os.path.join("..", "output", folder + ".csv") # <- recheck this
    new_df.to_csv(outpath, index=False)
    return new_df

def seasons(new_df): # split the dataframe into seasons
    s1 = data[data['Season'] == 'Season 1']
    s2 = data[data['Season'] == 'Season 2']
    s3 = data[data['Season'] == 'Season 3']
    s4 = data[data['Season'] == 'Season 4']
    s5 = data[data['Season'] == 'Season 5']
    s6 = data[data['Season'] == 'Season 6']
    s7 = data[data['Season'] == 'Season 7']
    s8 = data[data['Season'] == 'Season 8']
    return s1, s2, s3, s4, s5, s6, s7, s8

def count_emotions_s1(s1): # count the frequency of each emotion label for season 1
    # create a counter for each emotion label
    anger_count_s1 = 0
    disgust_count_s1 = 0
    fear_count_s1 = 0
    joy_count_s1 = 0
    neutral_count_s1 = 0
    sadness_count_s1 = 0
    surprise_count_s1 = 0

    for emotion in s1['Emotion_label']: #add 1 to the counter which corresponds to the emotion
        if emotion == "anger":
            anger_count_s1 += 1
        elif emotion == "disgust":
            disgust_count_s1 += 1
        elif emotion == "fear":
            fear_count_s1 += 1
        elif emotion == "joy":
            joy_count_s1 += 1
        elif emotion == "neutral":
            neutral_count_s1 += 1
        elif emotion == "sadness":
            sadness_count_s1 += 1
        elif emotion == "surprise":
            surprise_count_s1 += 1
    
    emotion_freq_s1 = [] # add every counter to a single list
    emotion_freq.append(anger_count_s1)
    emotion_freq.append(disgust_count_s1)
    emotion_freq.append(fear_count_s1)
    emotion_freq.append(joy_count_s1)
    emotion_freq.append(neutral_count_s1)
    emotion_freq.append(sadness_count_s1)
    emotion_freq.append(surprise_count_s1)
    return emotion_freq_s1, anger_count_s1, disgust_count_s1, fear_count_s1, joy_count_s1, neutral_count_s1, sadness_count_s1, surprise_count_s1
    
def emotion_list(): # create a list of all 7 emotion labels for the plots
    emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    return emotion_labels

def plot_seasons(emotion_labels, emotion_freq): # make a plot for every season with the distribution of emotions
    # season 1
    plt.figure(figsize = (16,6))
    plt.hist(emotions, emotion_freq_s1)
    plt.title('Season 1')
    plt.xlabel('Emotion labels')
    plt.ylabel('Frequencies')

def calculate_anger(anger_count_s1, anger_count_s2, anger_count_s3, anger_count_s4, anger_count_s5, anger_count_s6, anger_count_s7, anger_count_s8): # calculate the relative frequency of anger across seasons
    anger_total = anger_count_s1 + anger_count_s2 + anger_count_s3 + anger_count_s4 + anger_count_s5 + anger_count_s6 + anger_count_s7 + anger_count_s8
    rel_freq_anger_s1 = anger_count_s1/anger_total
    rel_freq_anger_s2 = anger_count_s1/anger_total
    rel_freq_anger_s3 = anger_count_s1/anger_total
    rel_freq_anger_s4 = anger_count_s1/anger_total
    rel_freq_anger_s5 = anger_count_s1/anger_total
    rel_freq_anger_s6 = anger_count_s1/anger_total
    rel_freq_anger_s7 = anger_count_s1/anger_total
    rel_freq_anger_s8 = anger_count_s1/anger_total
    # create a list with all relative frequencies
    anger_rel_freq = [rel_freq_anger_s1, rel_freq_anger_s2, rel_freq_anger_s3, rel_freq_anger_s4, rel_freq_anger_s5, rel_freq_anger_s6, rel_freq_anger_s7, rel_freq_anger_s8]
    return anger_freq

def season_list(): # create list of all seasons for the plots
    seasons = ["Season 1", "Season 2", "Season 3", "Season 4", "Season 5", "Season 6", "Season 7", "Season 8"]
    return seasons

def plot_rel_freq(anger_freq, disgust_freq, fear_freq, joy_freq, neutral_freq, sadness_freq, surprise_freq): # make a plot for every emotion with the distribution across seasons
    # anger
    plt.figure(figsize = (16,6))
    plt.hist(s1["Emotion_label"], bins = 7)
    plt.title('Anger')
    plt.xlabel('Seasons')
    plt.ylabel('Relative frequency')

if __name__=="__main__":
    classifier = load_classifier()
    data = load_data()
    scores = get_scores(data, classifier)
    emotions = get_labels(scores)
    new_df = update_dataframe(emotions)
    s1, s2, s3, s4, s5, s6, s7, s8 = seasons(new_df)
    emotion_freq_s1, anger_count_s1, disgust_count_s1, fear_count_s1, joy_count_s1, neutral_count_s1, sadness_count_s1, surprise_count_s1 = count_emotions_s1(s1)
    emotion_labels = emotion_list()
    anger_rel_freq = calculate_anger(anger_count_s1, anger_count_s2, anger_count_s3, anger_count_s4, anger_count_s5, anger_count_s6, anger_count_s7, anger_count_s8)
    seasons = season_list()
    plot_rel_freq(anger_freq, disgust_freq, fear_freq, joy_freq, neutral_freq, sadness_freq, surprise_freq)

#- Predict emotion scores for all lines in the data
#- For each season
#    - Plot the distribution of all emotion labels in that season
#    - restructure so u get counts for each season
#- For each emotion label
#    - Plot the relative frequency of each emotion across all seasons