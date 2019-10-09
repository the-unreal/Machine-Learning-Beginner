#Description: Build a movie recommendation engine (more specifically a content based recommendation engine)

#Import Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Load the data
from google.colab import files # Use to load data on Google Colab
uploaded = files.upload() # Use to load data on Google Colab
df = pd.read_csv("movie_dataset.csv")

#Print the first 3 rows of the data set
df.head(3)

#Get a count of the number of rows/movies in the data set and the number of columns
df.shape

#Create a list of important columns to keep a.k.a. the main content of the movie
features = ['keywords','cast','genres','director']

df[features].head(3)

#Clean and preprocess the data
for feature in features:
    df[feature] = df[feature].fillna('') #Fill any missing values with the empty string
   # print(df[feature])

#A function to combine the values of the important columns into a single string
def combine_features(row):
    return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]

#Apply the function to each row in the dataset to store the combined strings into a new column called combined_features 
df["combined_features"] = df.apply(combine_features,axis=1)
#df["combined_features"]

#Print the data frame to show the new column 'combined_features'
df.head(3)

#Convert a collection of text to a matrix/vector of token counts
count_matrix = CountVectorizer().fit_transform(df["combined_features"])

#Print the count matrix
#print(count_matrix.toarray())

#Get the cosine similarity matrix from the count matrix (cos(theta))
cosine_sim = cosine_similarity(count_matrix)

#Print the cosine similarity matrix
print(cosine_sim)

#Get the number of rows and columns in the data set
cosine_sim.shape

#Helper function to get the title from the index
def get_title_from_index(index):
  return df[df.index == index]["title"].values[0]

#Helper function to get the index from the title
def get_index_from_title(title):
  return df[df.title == title]["index"].values[0]

#Get the title of the movie that the user likes
movie_user_likes = "The Amazing Spider-Man"

#Find that movies index
movie_index = get_index_from_title(movie_user_likes)

#Access the row, through the movies index, corresponding to this movie (the liked movie) in the similarity matrix, 
# by doing this we will get the similarity scores of all other movies from the current movie

#Enumerate through all the similarity scores of that movie to make a tuple of movie index and similarity scores.
#  This will convert a row of similarity scores like this- [5 0.6 0.3 0.9] to this- [(0, 5) (1, 0.6) (2, 0.3) (3, 0.9)] . 
#  Note this puts each item in the list in this form (movie index, similarity score)
similar_movies =  list(enumerate(cosine_sim[movie_index]))



#Sort the list of similar movies according to the similarity scores in descending order
#Since the most similar movie is itself, we will discard the first element after sorting.
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

#Print the sorted similar movies to the movie the user like
# The tuples are in the form (movie_index, similarity value)
print(sorted_similar_movies)

#Create a loop to print the first 5 entries from the sorted similar movies list

i=0
print("Top 5 similar movies to "+movie_user_likes+" are:")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]) )
    i=i+1
    if i>=5:
        break

#Create a loop to print the first 5 entries from the sorted similar movies list 
# and similarity scores

i=0
print("Top 5 similar movies to "+movie_user_likes+" are:")
for i in range( len(sorted_similar_movies)):
    print('Movie title:',get_title_from_index(sorted_similar_movies[i][0]), ', Similarity Score: ', sorted_similar_movies[i][1] )
    i=i+1
    if i>=5:
        break
