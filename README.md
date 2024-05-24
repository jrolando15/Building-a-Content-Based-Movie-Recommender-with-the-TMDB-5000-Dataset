# Building-a-Content-Based-Movie-Recommender-with-the-TMDB-5000-Dataset

# Project Description
This project implements a content-based movie recommender system using the TMDB 5000 dataset. The system recommends movies based on the similarity of their content, such as genres, keywords, cast, and crew.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [License](#license)

# Introduction
The goal of this project is to build a content-based movie recommender system using the TMDB 5000 dataset. The system uses natural language processing techniques to analyze movie metadata and recommend similar movies based on their content.

# Installation
To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- scikit-learn
- nltk

You can install the required libraries using the following command:
```bash
pip install pandas numpy scikit-learn nltk
```

# Usage
1. Clone the repository
```bash
git clone https://github.com/your_username/Content-Based-Movie-Recommender.git
cd Content-Based-Movie-Recommender
```

2. Run the Script
```python
RS_model.ipynb
```

# Project Structure
```bash
Content-Based-Movie-Recommender/
├── dataset/
│   ├── tmdb_5000_credits.csv          # Credits dataset
│   ├── tmdb_5000_movies.csv           # Movies dataset
├── recommender.ipynb                   # Main script with the code
├── README.md                          # Project README file
```

# Data Processing
The dataset is loaded using pandas and merged on the 'title' column. Missing values are handled by dropping rows with missing data. The following steps are performed:
- Parsing JSON-like columns to extract relevant information.
- Selecting the top 5 cast members and the director.
- Removing spaces from the extracted features.
- Combining genres, keywords, cast, and crew into a single 'tags' column.
- Applying stemming to the 'tags' column using the PorterStemmer from nltk.

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load datasets
credits = pd.read_csv('dataset/tmdb_5000_credits.csv')
movies = pd.read_csv('dataset/tmdb_5000_movies.csv')

# Merge datasets on the 'title' column
movies['title'] = movies['title'].astype(str)
credits['title'] = credits['title'].astype(str)
merged_df = movies.merge(credits, on='title')

# Select relevant columns
merged_df = merged_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
merged_df.dropna(inplace=True)

# Parse JSON-like columns
def parse_list(obj):
    return [item['name'] for item in ast.literal_eval(obj)]

merged_df['genres'] = merged_df['genres'].apply(parse_list)
merged_df['keywords'] = merged_df['keywords'].apply(parse_list)

# Get top 5 cast members
def get_top_cast(obj):
    return [item['name'] for item in ast.literal_eval(obj)[:5]]

merged_df['cast'] = merged_df['cast'].apply(get_top_cast)

# Get director
def get_director(obj):
    for item in ast.literal_eval(obj):
        if item['job'] == 'Director':
            return [item['name']]
    return []

merged_df['crew'] = merged_df['crew'].apply(get_director)

# Remove spaces
def remove_spaces(L):
    return [item.replace(' ', '') for item in L]

merged_df['cast'] = merged_df['cast'].apply(remove_spaces)
merged_df['crew'] = merged_df['crew'].apply(remove_spaces)
merged_df['keywords'] = merged_df['keywords'].apply(remove_spaces)
merged_df['genres'] = merged_df['genres'].apply(remove_spaces)

# Combine features into a single 'tags' column
merged_df['tags'] = merged_df['genres'] + merged_df['keywords'] + merged_df['cast'] + merged_df['crew']
merged_df['tags'] = merged_df['tags'].apply(lambda x: " ".join(x).lower())

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to stem words
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])

# Apply stemming to the 'tags' column
merged_df['tags'] = merged_df['tags'].apply(stem_words)
```

# Model Training 
The CountVectorizer from scikit-learn is used to convert the 'tags' column into a matrix of token counts. Cosine similarity is then calculated between the vectors to measure the similarity between movies.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vectorize the 'tags' column
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(merged_df['tags']).toarray()

# Calculate cosine similarity
similarity_matrix = cosine_similarity(vectors)

# Function to recommend movies
def recommend_movie(movie_title):
    if movie_title not in merged_df['title'].values:
        print("Movie not Found in the dataset.")
        return
    movie_index = merged_df[merged_df['title'] == movie_title].index
    distances = similarity_matrix[movie_index]
    movie_indices = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    recommendations = [merged_df.iloc[i].title for i in movie_indices]
    print("Recommendations for '{}':".format(movie_title))
    for title in recommendations:
        print(title)

if __name__ == "__main__":
    user_movie = input("Enter a movie name to get recommendations: ")
    recommend_movie(user_movie)
```

# Model Evaluation
The recommender system is evaluated by manually checking the recommendations for a given movie title. The system prints the top 5 recommended movies based on the similarity scores.

# License 
This README file provides a comprehensive overview of the project, including installation instructions, usage, project structure, data processing, model training, and evaluation. It also includes a license section to specify the project's licensing terms.

