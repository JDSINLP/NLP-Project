# GitHub Bitcoin README 


# Project Description
The goal of the project is to web scrape Github Bitcon repository and use README files to build a model to predict programming language of a repository.


# The Plan

* Acquire data
    * acquired data by scraping github repos README using the acquire.py script and saved locally as json file.
    
* Prepare data
     - bin language into 'Python', 'JavaScript', 'C++', 'Java','Other'
     - convert words to lower case 
     - Remove any accented characters, non-ASCII characters
     - Remove special characters.
     - Lemmatize the words.
     - store the clean text into a column named readme_contetns_clean
     - add columns 
         - readme_contents_clean: contains cleaned readme
         - length: lenght of clean readme
         - unique: number of unique words in clean readme
     - split data into train, val, and test(approx. 60/20/20)

* Explore Data
    * Use graph to explore data
        * What are the most common words in READMEs?
        * Does the length of the README vary by programming language?
        * Do different programming languages use a different number of unique words?
       
* Develop Model
    * Isolate a target variables
    * Set up baseline prediction
    * Evaluate models on train data and validate data
    * Select the best model based on the highest accuracy 
    * Evaluate the best model on test data to make predictions

* Draw Conclusions

# Data Dictionary
| Feature | Definition |
|:--------|:-----------|
| repo| the name of repository|
| language| the progamming language|
| readme_contents| text of README|
| readme_contents_clean|clean text of README|
| length| lenght of README|
| Unique| count of unique words in clean text of README|

# Steps to Reproduce
1. Clone this repo  
2. Run notebook

# Takeaways and Conclusions
    
* 
* 
* 
* 
* 

# Recommendations & Next Steps
* 