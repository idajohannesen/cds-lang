# Assignment 2

# Short description:
This code trains two classification models on a fake news dataset. The code is split into three scripts with a function each: the first is a vectorizer, the second is a logistic regression classifier, and the third is a neural network classifier. All three functions can be found in the ```src``` folder. Models are saved to the ```models``` folder and the ```output``` folder includes classification reports, performance visualizations, and predictions for test sentences.

The logistic regression classifier prints the 20 most informative features, a classification report, a confusion matrix, graphs for cross validation, and the final prediction for the test sentence.

The neural network classifier prints a classification report, a confusion matrix, a loss curve plot, and the final prediction for the test sentence.

This code uses CodeCarbon to track CO₂ emissions. The results are discussed in Assignment 5.

# Data:
The dataset is a corpus of real and fake news. It can be inserted as a csv file in the ```input``` folder. 
The data can be downloaded from here: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news

# Reproducing:
A setup file has been included which creates a virtual environment with the necessary requirements, as well as a ```run.sh``` file which activates the virtual environment and runs the script. The script requires that the data is located in the ```input``` folder with the structure mentioned above. 

Start by running the setup file by inputting ```bash setup.sh``` in the terminal. 
The code can then be run through the terminal by inputting ```bash run.sh```. This runs all three scripts.
Both of these lines should be executed from the ```assignment 1``` folder.

The ```src``` folder also includes a necessary utils script which does not have to be run manually.
The code has been written and tested on a Windows laptop and may not work correctly on other operating systems.

# Discussion/summary:
Both models run with an 89% accuracy score, with the mlp classifier having slightly more varied recall and precision scores than the logistic regression. It should be noted, however, that the models cannot account for misspellings or tokenization errors, and they may output an unexpected result if these are present in the texts. 