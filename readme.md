# GitHub Bitcoin README 


# Project Description
The goal of the project is to scrape Github Bitcoin repositories and build a model to predict programming language of a repository using the natural language content in README files. A link to summarized Canva slides can be found here: https://www.canva.com/design/DAFccDwPXcA/N2HuzVkjXTmxmMnMrd0jJg/edit?utm_content=DAFccDwPXcA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


# The Plan

* Acquire data
    * acquired data as of March 3, 2023 by scraping github repos README using the acquire.py script and saved locally as json file.
    
* Prepare data
     - bin language into 'Python', 'JavaScript', 'C++', 'Java','Other'
     - convert words to lower case 
     - Remove any accented characters, non-ASCII characters
     - Remove special characters.
     - Lemmatize the words.
     - store the clean text into a column named readme_contents_clean
     - add columns 
         - readme_contents_clean: contains cleaned readme
         - length: length of clean readme
         - unique: number of unique words in clean readme
     - split data into train, val, and test(approx. 60/20/20)

* Explore Data
    * Use graphs to explore data
        * What are the most common words in READMEs?
        * Does the length of the README vary by programming language?
        * Do different programming languages use a different number of unique words?
       
* Develop Model
    * Isolate target variable
    * Establish baseline
    * Evaluate models on train and validate sets
    * Select the best model based on the highest accuracy 
    * Run the best model on test data to make predictions

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
2. Update env.py file with github_token and github_username
3. Run Notebook

# Takeaways and Conclusions
    
* 'http', 'github com', and 'http github com' were the top unigrams, bigrams and trigams respectively
* JavaScript had the longest README
* Java had most unique word count
* Bitcoin was not a most common words
* Decision Tree with max_depth 4 has accuracy score of 65%  on test data beating baseline accuracy score of 39% by  26 %

# Recommendations & Next Steps
* Web-scraping takes time, so be patient
* Scrape many repos to get lots of content
* With more time we would scrape more repos, try stemming instead of lemmatizing, and add more stop-words
