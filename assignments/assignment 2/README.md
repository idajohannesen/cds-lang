This code trains two classification models on a fake news dataset. The notebook is split into three sections with a function each: the first is a vectorizer, the second is a logistic regression, and the third is a neural network. All three functions can be found in the assignment2.py script. 

The logistic regression classifier prints the 20 most informative features, a classification report, a confusion matrix, graphs for cross validation, and the final prediction for the test sentence.

The neural network classifier prints a classification report, a confusion matrix, a loss curve plot, and the final prediction for the test sentence.

Each of the models' classification reports are saved as text files in the 'output' folder. The trained models and the vectorizer are saved in the 'models' folder. The reports and models are saved as part of their respective scripts.

Setup requires running the setup.sh script

sorry about the frankenstein graph situation going on with the neural network's loss curve and confusion matrix. i have no clue what happened or how to fix it