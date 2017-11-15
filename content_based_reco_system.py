# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    
    # Get the genre column from the dataframe & create a list of list with tokenization output
    res=[]
    
    genre=movies['genres']
    for e in genre:
        token=e
        res_tmp=tokenize_string(token)
        res.append(res_tmp)
    
    # Create a new column in the movies dataframe  using the above list of lists
    length = len(movies['genres'])
    movies.loc[:,'tokens'] = pd.Series(res, index=movies.index)
    
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    # Get the list of genres for all movies
    nmov = movies.movieId.nunique()
    tokens=list(movies['tokens'])
    
    # Calculate number of unique documents of each of the genre
    feats=[j for i in tokens for j in i]
    feats_cnt=dict(Counter(feats))
    res=[]
    
    # Looping through movies
    for each in tokens:
        
        # Get frequency of all genres in this movie (to see if there are repeated genres)
        cnt=dict(Counter(each))
        
        # Number of features for this document
        n_feats=len(cnt)
        
        # Genre frequency for genre with maximum frequency
        max_d=cnt[max(cnt,key=cnt.get)]
            
        # Create the TF & IDF score for each genre
        for key, value in cnt.items():
            df_i=feats_cnt[key]
            cnt[key] = (value / max_d) * math.log10(nmov/df_i)
        res.append(cnt)
        
    # Create vocab dictionary with the genre & column indices for each genre in alphabetic order
    col_set=list(feats_cnt)
    col_set=sorted(col_set)

    vocab=[]
    i=-1
    for each in col_set:
        i+=1
        vocab.append((each,i))
    vocab_final=dict(vocab)
    
    # Creating CSR matrix
    ncol=len(vocab_final)
    nrow=1
    
    mat_res=[]
    
    # Loop through each movie from the res dictionary
    for each in res:
        rows=[]
        cols=[]
        val=[]
        row=0
        # Loop through each genre of the movie
        for e in each.items():
            row_token=row
            rows.append(row_token)
        
            col_token=vocab_final[e[0]]
            cols.append(col_token)
           
            val_tmp=e[1]
            val.append(val_tmp)
        
        row_final = np.array(rows)
        col_final = np.array(cols)
        data = np.array(val)
        res_matrix=csr_matrix((data, (row_final, col_final)), shape=(nrow,ncol))
        mat_res.append(res_matrix)
        
    # Storing the CSR matrix as a feature in the dataframe
    length = len(movies['tokens'])
    movies.loc[:,'features'] = pd.Series(mat_res, index=movies.index)
        
    return movies,vocab_final

def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]

def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    
    am=a.toarray()[0]
    bm=b.toarray()[0]
    
    nmrtr=np.dot(am,bm)
    dnmtr=np.sqrt(np.dot(am,am))*np.sqrt(np.dot(bm,bm))
    return nmrtr/dnmtr

def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """

    # Get all the movies in test dataframe
    tmovies=ratings_test['movieId'].unique()
    
    # Get the information of CSR matrix from main movie data frame for the test & train movies
    tmovies_all=movies[movies['movieId'].isin(tmovies)]
    trmovies_all=movies[-movies['movieId'].isin(tmovies)]
    
    sim_movies={}
        
    # Create a master similarity list for every test movie with every train movie when similarity>0
    for u,v in tmovies_all.iterrows():
        cur_m=v.movieId
        cur_csr=v.features
           
        # Loop through every other movie to obtain similarity
        for x,y in trmovies_all.iterrows():
                nex_m=y.movieId
                nex_csr=y.features
                mtmp=sorted(tuple((cur_m,nex_m)))
                s=str(mtmp[0])+','+str(mtmp[1])
                sim=cosine_sim(cur_csr,nex_csr)
                if sim>0:
                    sim_movies.update({s:sim})
    
    
    result=[]
    
    # Create a list of lists from the above list of tuples keeping only the movie IDs
    # For every user & movie combination in the test data
    for p,q in ratings_test.iterrows():
        tuser=int(q.userId)
        tmovie=int(q.movieId)
        
        # Get the list of movies from training data that this user has rated and movies which have a similarity>0
        umovies_tmp=ratings_train[ratings_train.userId==tuser]['movieId']
        
        # Create a list of lists where each sublist is a tuple of the movie i and one movie that this user rated
        stmp_f=[]
        stmp_f1=[]
        
        for e in umovies_tmp:
            stmp=sorted(tuple((tmovie,e)))
            t=str(stmp[0])+','+str(stmp[1])
            stmp_f.append(t)
            stmp_f1.append(stmp)
           
        # Get the subset of ratings list for movies present above
        # Get subset of movies that are similar to current movie
        mlist_f=[]
        for e in stmp_f1:
            if (str(e[0])+','+str(e[1])) in sim_movies.keys():
                mlist_f.append(e)     
        clist=[j for i in mlist_f for j in i]   
        clist_f=[c for c in clist if c!=tmovie]
        
        # Filter the user ratings data for the above movies
        umovies1=ratings_train[ratings_train.userId==tuser]
        umovies2=umovies1[umovies1['movieId'].isin(clist_f)]
         
        shp=umovies2.shape[0]
        umovies3=umovies2.copy()
            
        # If we find similar movies then
        if shp>0:
        
            sim_f=[]
            # Add the similarity score as a column in the above dataframe
            for i,r in umovies2.iterrows():
                g=int(r.movieId)
                stmp=sorted(tuple((tmovie,g)))
                t=str(stmp[0])+','+str(stmp[1])
                sim=sim_movies[t]
                sim_f.append(sim)
        
                
            umovies3['sim_score'] = pd.Series(sim_f, index=umovies3.index)
            
            # Calculating weighted average for each user for the movie
            wght=np.array(umovies3.sim_score)
            rate=np.array(umovies3.rating)
            
            nmrtr=np.dot(wght,rate)
            dnmtr=np.sum(wght)
            
            pred=nmrtr/dnmtr
            result.append(pred)
            
        else:
            
            # Add the regular average of user's ratings for all movies
            rate=np.array(umovies1.rating)
            
            nmrtr=np.sum(rate)
            dnmtr=len(umovies1.movieId)
            
            pred=nmrtr/dnmtr
            result.append(pred)
    
    res=np.array(result)
    return res


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
