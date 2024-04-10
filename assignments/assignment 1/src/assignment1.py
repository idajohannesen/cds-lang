import spacy
import pandas as pd
import os
import re

def load_model(): # load the model using spacy
    nlp = spacy.load("en_core_web_md")
    return nlp

def create_path(): # create path to the input directory
    main_folder_path = "../input/USEcorpus/"
    return main_folder_path

def sort_directory(main_folder_path): # sort the directory
    sorted_dir = sorted(os.listdir(main_folder_path))
    return sorted_dir

def extract_features(nlp, main_folder_path, sorted_dir): # creating a for loop going into each text file within the subfolders and extracting linguistic features
    for folder in sorted_dir:
        folder_path = os.path.join(main_folder_path, folder)
        filenames = sorted(os.listdir(folder_path))
        # creating a list for every folder's info to be gathered in. will be used to create dataframes later
        folder_info = []
    
        for text_file in filenames:
            file_path = folder_path + "/" + text_file
            
            with open (file_path, encoding="latin-1") as file:
                text = file.read()
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
            outpath = os.path.join("..", "output", folder + ".csv")
            df.to_csv(outpath, index=False)

def main():
    nlp = load_model()
    main_folder_path = create_path()
    sorted_dir = sort_directory(main_folder_path)
    extract_features(nlp, main_folder_path, sorted_dir)

if __name__=="__main__":
    main()