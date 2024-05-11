from codecarbon import EmissionsTracker
import os
# CodeCarbon tracker
outfolder = os.path.join("..", "assignment 5", "input") # set output folder
tracker = EmissionsTracker(project_name="assignment 1", # create tracker
                           experiment_id="linguistic-feature-extraction",
                           output_dir=outfolder,
                           output_file="emissions.csv")
tracker.start_task("load packages")

# loading model
import spacy

# creating dataframes
import pandas as pd

# visualization
import matplotlib.pyplot as plt

# removing metadata from files
import re

tracker.stop_task()

def load_model():
    """
    Loads the model using spacy

    Returns:
        nlp: language model
    """
    nlp = spacy.load("en_core_web_md")
    return nlp

def create_path():
    """
    Creates path to the input directory

    Returns: 
        main_folder_path: path to the dataset
    """
    main_folder_path = "input/USEcorpus/"
    return main_folder_path

def sort_directory(main_folder_path):
    """
    Sorts the directory

    Arguments:
        main_folder_path: path to the dataset
    Returns:
        sorted_dir: the dataset, sorted
    """
    sorted_dir = sorted(os.listdir(main_folder_path))
    return sorted_dir

def extract_features(nlp, main_folder_path, sorted_dir):
    """
    Creates a for loop going into each text file within the subfolders and extracting linguistic features

    Arguments:
        nlp: language model
        main_folder_path: path to the dataset
        sorted_dir: the dataset, sorted
    Returns:
        column_names: names of columns in csv files
    """
    for folder in sorted_dir:
        folder_path = os.path.join(main_folder_path, folder)
        filenames = sorted(os.listdir(folder_path))
        # creating a list for every folder's info to be gathered in. will be used to create dataframes later
        folder_info = []
    
        for text_file in filenames:
            file_path = folder_path + "/" + text_file
            
            with open (file_path, encoding="latin-1") as file:
                text = file.read()
                text = text.lower() # make everything lowercase
                # removing the metadata and document ID inside <>
                text = re.sub(r'<.+?>', '', text)
                # add text files to a doc object
                doc = nlp(text)
                # create counters for nouns, verbs, adjectives, and adverbs.
                noun_count = 0
                verb_count = 0
                adj_count = 0
                adv_count = 0

                # with a for loop, add 1 to the counter every time that part of speech appears in the doc object
                for token in doc:
                    if token.pos_ == "NOUN":
                        noun_count += 1
                    elif token.pos_ == "VERB":
                        verb_count += 1
                    elif token.pos_ == "ADJ":
                        adj_count += 1
                    elif token.pos_ == "ADV":
                        adv_count += 1
                
                # relative frequencies of nouns, verbs, adjectives, and adverbs per 10,000 words, rounded up to 2 decimals        
                relative_freq_noun = round((noun_count/len(doc)) * 10000, 2)
                relative_freq_verb = round((verb_count/len(doc)) * 10000, 2)
                relative_freq_adj = round((adj_count/len(doc)) * 10000, 2)
                relative_freq_adv = round((adv_count/len(doc)) * 10000, 2)
                
                # extracting named entities for persons, locations, and organizations
                persons = set()
                for ent in doc.ents:
                        if ent.label_ == 'PERSON':
                            persons.add(ent.text)
                num_persons = len(persons)
                
                locations = set()
                for ent in doc.ents:
                        if ent.label_ == 'LOC':
                            locations.add(ent.text)
                num_locations = len(locations)
                
                organisations = set()
                for ent in doc.ents:
                        if ent.label_ == 'ORG':
                            organisations.add(ent.text)
                num_organisations = len(organisations)
                
                # create list for every file
                file_info = [text_file, relative_freq_noun, relative_freq_verb, relative_freq_adj, relative_freq_adv, num_persons, num_locations, num_organisations]
                # append the file's info to the collected list for the whole folder's info
                folder_info.append(file_info)
        
            # creating a dataframe with pandas using folder_info
            # make one per subfolder
            
            df = pd.DataFrame(folder_info,
                        columns=["Filename", "RelFreq NOUN", "RelFreq VERB", "RelFreq ADJ", "RelFreq ADV", "Unique PER", "Unique LOC", "Unique ORG"])
            
            # upload dataframe to output folder
            outpath = os.path.join("output", folder + ".csv")
            df.to_csv(outpath, index=False)

def calculate_averages():
    """
    Calculates the average score of each linguistic feature for every csv file

    Arguments:
    Returns:
        averages: csv file of averages for every feature
        column_names: names of columns in csv files
    """
    all_files = [] # create empty list for the averages of each csv file to go into
    column_names = ["RelFreq NOUN", "RelFreq VERB", "RelFreq ADJ", "RelFreq ADV", "Unique PER", "Unique LOC", "Unique ORG"] # list of column names
    sorted_outputs = sorted(os.listdir("output/")) # sorting the output directory
    for csv in sorted_outputs:
        if csv.endswith(".csv"): # only load csv files
            current_file = pd.read_csv("output/" + csv)
            file_info = [] # create a list for the current file
            for column in column_names:
                total = sum(current_file[column]) # finds the sum of results
                length = len(current_file) # finds the amount of files
                average = round(total/length, 1)
                
                file_info.append(average) # add averages
            all_files.append(file_info)

    # list of every csv file's name
    csv_list = ["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "c1"]
    averages = pd.DataFrame(all_files, columns=column_names, index = csv_list) # creating a dataframe for every newspaper using pandas

    # upload dataframe to output folder
    averages.to_csv("output/average/averages.csv", index=False)
    return averages, column_names

def plot(averages, column_names):
    """
    Plots the average score of each linguistic feature for every csv file

    Arguments:
        averages: csv file of averages for every feature
        column_names: names of columns in csv files
    """

    plt.figure(figsize=(12,6))
    plt.plot(averages)
    plt.legend(column_names, loc='upper left')
    plt.title("Linguistic features over time")
    plt.xlabel("Filenames")
    plt.ylabel("Linguistic features")

    plt.savefig('output/average/plot.png') # save output

def main():
    tracker.start_task("load model")
    nlp = load_model()
    tracker.stop_task()

    tracker.start_task("create path to dir")
    main_folder_path = create_path()
    tracker.stop_task()

    tracker.start_task("sort directory")
    sorted_dir = sort_directory(main_folder_path)
    tracker.stop_task()

    tracker.start_task("extract features")
    extract_features(nlp, main_folder_path, sorted_dir)
    tracker.stop_task()

    tracker.start_task("calculate averages")
    averages, column_names = calculate_averages()
    tracker.stop_task()

    tracker.start_task("plotting")
    plot(averages, column_names)
    tracker.stop_task()

    tracker.stop()

if __name__=="__main__":
    main()