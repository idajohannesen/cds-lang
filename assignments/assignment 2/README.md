# Assignment 2

# Short description:
This code trains two classification models on a fake news dataset which attempts to predict whether an article is real or fake news. This is demonstrated with a test sentence at the end of the scripts. The code is split into three scripts with a function each: the first is a vectorizer, the second is a logistic regression classifier, and the third is a neural network classifier. All three functions can be found in the ```src``` folder. Models are saved to the ```models``` folder and the ```output``` folder includes classification reports, performance visualizations, and predictions for test sentences.

The logistic regression classifier prints the 20 most informative features, a classification report, a confusion matrix, graphs for cross validation, and the final prediction for the test sentence.

The neural network classifier prints a classification report, a confusion matrix, a loss curve plot, and the final prediction for the test sentence.

This code uses CodeCarbon to track CO₂ emissions. The results are discussed in Assignment 5.

# Data:
The dataset is a corpus of articles containing real and fake news.
The data can be downloaded from here: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news
Download the data from the link and unzip it. Inside the unzipped folder, there is a csv file titled ```fake_or_real_news.csv```. Place this csv file in the ```input``` folder.

# Reproducing:
A setup file has been included which creates a virtual environment with the necessary requirements, as well as a ```run.sh``` file which activates the virtual environment and runs the script. The script requires that the data is located in the ```input``` folder with the structure mentioned above. 

Start by running the setup file by inputting ```bash setup.sh``` in the terminal. 
The code can then be run through the terminal by inputting ```bash run.sh```. This runs all three scripts.
Both of these lines should be executed from the ```assignment 1``` folder.

The ```src``` folder also includes a necessary utils script which does not have to be run manually.
The code has been written and tested on a Windows laptop and may not work correctly on other operating systems.

# Discussion/summary:
The logistic classifier model has an accuracy score of 89%, while the mlp classifier ends up at 90%. These are robust scores as a seed has been set for both models. Changing the training/test split to 90/10 did not affect the accuracy for either model, but the mlp classifier was improved from 89% to the final 90% by changing to use a relu activation instead of a logistic activation for the hidden layer. 
It should be noted, however, that the models cannot account for misspellings or tokenization errors, and they may output an unexpected result if these are present in the texts. For example, the test sentence includes the name Hillary, which is an indicator of fake news to the model. If this name is misspelled, for example as 'Hilary', the model will classify the test sentence as real news instead, as the different spelling is not a feature associated with fake news. Small issues such as misspellings or a wrong tokenization can therefore change the classification, although this is likely much more impactful for shorter validation tests, such as the single test sentence we offer the models, rather than longer articles which would have more data to base the classification off of. 