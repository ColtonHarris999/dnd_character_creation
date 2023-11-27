# D&D Character Creation Analysis

This project is centered around performing analysis on Dungeons and Dragons character data.

# Libraries to install
geopandas
matplotlib
pandas
sklearn
torch

# Datasets
There are two data folders.
Both should be included with this project's submission, I reccomend using these
rather than downloading the datasets using the links in the report.
### data
This is where the character data and geospatial data used to run the main program is.
Note you will also need this folder to run the test program.
### test_data
This is the folder containing subsets of character data used only for the testing program.

# Runnable Files
Both runable files will print "Done!" to the terminal when finished.
### dnd_analysis.py
This is the main program. Will save 7 images to the same folder this file is in.
Takes about a mintue to run and will print a few things to the console:
1. The current loss of that pass through of the data while building the nueral net
2. The accuracy of the network on the traning data
3. The accuracy of the network on the testing data
4. "Done!"  
### dnd_test.py
This is the testing program which runs independently of dnd_analysis.py.
It will save 5 images to the folder the file is in.
Prints "Done!" after tests are ran.
# All Python Files
### dnd_analysis.py
This is the runnable main program used to perform all of the analysis.
Creates 7 different plots of character data, first analyzing national data and
then analyzing the character's stats. Creates a neural network to predict
a character's class given their stats and plots the accuracy of the model.
See **Runable Files** section for more info.
### data_setup.py
This file is called to initialize data and setup the overall analysis.
There are functions to get:
1. Character data
2. Map data
3. Character & Map data 
4. Character's weapons and player's country
5. Tensor data for training the network
### characters_by_nationality
This file is focused on Q1 of the report and plots the different
graphs analyzing nationality and different character creation aspects.
### class_prediction.py
This file is centered around Q2 and Q3. It is where the network is constructed
and trained. There are also functions to plot the balance of the data and the
accuracy/inaccuracy of the model.
### dnd_test.py
This is the runnable testing file. It tests most of the ploting functions
and all of the data setup functions. Note that it needs both data folders,
not just the testing data folder. Will print "Done!" to the terminal when
finished running. See **Runable Files** section for more info.
### cse163.utils.py
This is the standard CSE 163 utilities folder used for testing.
The assert_equals function used for testing resides in this file.

