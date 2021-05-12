#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
import os
import csv
from tqdm import tqdm
import random
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import pymc3 as pm
import theano
theano.config.compute_test_value = 'raise'
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import scipy.stats as st
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial


# # Data Preprocessing

# In[5]:


best_movies_rating = pd.read_csv("C:\\Users\\17326\\Documents\\Data Science\\Artificial Intelligence\\Final Project\\best_movie_ratings_features.csv")
best_movies_rating.head()


# In[6]:


best_movies_rating.genres = best_movies_rating.genres.apply(lambda x:eval(x)[0])
best_movies_rating.year = best_movies_rating.year.apply(lambda x:eval(x)[0])
best_movies_rating.head()


# In[7]:


best_movies_rating.genres.unique()


# In[10]:


onehot_genres = pd.get_dummies(best_movies_rating.genres)
onehot_year = pd.get_dummies(best_movies_rating.year)


# In[12]:


Eng_Features = pd.concat([best_movies_rating[["title", "rating", "votes"]], onehot_genres, onehot_year], axis=1)
Eng_Features = Eng_Features.set_index("title")
Eng_Features.head()


# In[13]:


Eng_Features.to_csv("C:\\Users\\17326\\Documents\\Data Science\\Artificial Intelligence\\Final Project\\best_movie_ratings_Eng_features.csv")


# In[17]:


movies = pd.read_csv("C:\\Users\\17326\\Documents\\Data Science\\Artificial Intelligence\\Final Project\\best_movie_ratings_Eng_features.csv", index_col=0)
movies.rating = movies.rating/10
movies.head()


# In[18]:


users = pd.read_csv("C:\\Users\\17326\\Documents\\Data Science\\Artificial Intelligence\\Final Project\\users_ratings.csv", index_col=0)
users.rating = users.rating/10
users.head()


# ### Dimension Reduction

# In[99]:


WANTED_DIM = 20


# In[30]:


pca_df = movies[list(movies.columns[2:])]
pca = PCA(n_components=20)
pca_df = pd.DataFrame(pca.fit_transform(pca_df))
pca_df.index = movies.index

movies.shape


# In[32]:


#Varience in database
pca.explained_variance_ratio_.sum()


# In[33]:


movies = pd.concat([movies[list(movies.columns[:2])], pd.DataFrame(pca_df)] ,axis=1)


# In[85]:


collabo = movies.merge(users, left_index=True, right_index=True)

for n in range(20):
    collabo[n] = (collabo[n] * collabo['rating_x'])* collabo['rating_x'] 

collabo = collabo.groupby(collabo.user).aggregate(np.average)

for n in range(20):
    collabo[n] = (collabo[n] * collabo['rating_x'])

collabo = collabo[[n for n in range(20)]]

collabo.sample(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Learning

# In[86]:



class algorithm:
    def update_features(self, user_features, movie_features, rating, t):
        return update_features(user_features, movie_features, rating, t)
    def compute_utility(self, user_features, movie_features, epoch, s):
        return compute_utility(user_features, movie_features, epoch, s)
    
class random_choice(algorithm):
    def choice(self, user_features, movies, epoch, s):
        return movies.sample()


# In[87]:


def greedy_choice_t(user_features, movies, epoch, s, recommf):
    epsilon = 1 / math.sqrt(epoch+1)
    return greedy_choice_no_t(user_features, movies, epoch, s, recommf, epsilon)

def greedy_choice_no_t(user_features, movies, epoch, s, recommf, epsilon):
    if random.random() > epsilon:
        return recommf(user_features, movies, epoch, s)
    else:
        return movies.sample()
    
class greedy_choice_contentbased(algorithm):
    def choice(self, user_features, movies, epoch, s):
        return greedy_choice_t(user_features, movies, epoch, s, best_contentbased_recommandation)

class greedy_choice_no_t_contentbased(algorithm):
    def choice(self, user_features, movies, epoch, s, epsilon=0.3):
        return greedy_choice_no_t(user_features, movies, epoch, s, best_contentbased_recommandation, epsilon)
    
class greedy_choice_UCB(algorithm):
    def choice(self, user_features, movies, epoch, s):
        return greedy_choice_t(user_features, movies, epoch, s, partial(best_contentbased_recommandation, UCB=True))


# In[88]:


class greedy_choice_collaborative(algorithm):
    def choice(self, user_features, movies, epoch, s):
        return greedy_choice_t(user_features, movies, epoch, s, best_collaborative_recommandation)

class greedy_choice_no_t_collaborative(algorithm):
    def choice(self, user_features, movies, epoch, s, epsilon=0.3):
        return greedy_choice_no_t(user_features, movies, epoch, s, best_collaborative_recommandation, epsilon)


# In[89]:


class LinUCB(algorithm):
    def __init__(self, alpha):
        self.first = True
        self.alpha = alpha
        
    def choice(self, user_features, movies, epoch, s):
        # movies features
        x = movies.apply(get_movie_features, axis=1).as_matrix()
        # number of movies
        m = x.shape[0]
        # dimension of movie features
        d = x.shape[1]
        # initialize when first time
        if self.first:
            self.first = False
            self.A = np.zeros((m, d, d))
            for a in range(m):
                self.A[a] = np.eye(d)
            self.b = np.zeros((m, d))
        # get rating for every movie
        ratings = np.zeros(m)
        for a, (title, movie) in enumerate(movies.iterrows()):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv.dot(self.b[a])
            ratings[a] = theta_a.T.dot(x[a]) + self.alpha * np.sqrt(x[a].T.dot(A_inv).dot(x[a]))
        self.recomm = ratings.argmax()
        chosen = movies[movies.index == movies.index[self.recomm]]
        self.A[self.recomm] += x[self.recomm].dot(x[self.recomm].T)
        return chosen
    
    def update_features(self, user_features, movie_features, rating, t):
        self.b[self.recomm] += rating * movie_features
        return super().update_features(user_features, movie_features, rating, t)
    
    def compute_utility(self, user_features, movie_features, epoch, s):
        return user_features.dot(movie_features)


# In[90]:



def bayes_UCB(user_features, movies, epoch, s):
    # Hyperparameters
    c0 = 10
    d0 = 3
    e0 = 0.01
    f0 = 0.001
    g0 = 0.001
    # function
    I = np.eye(user_features.size)
    ratings = np.zeros(movies.shape[0])
    with pm.Model():
        s = pm.Gamma('s', d0, e0)
        sigma = pm.InverseGamma('sigma', f0, g0)
        theta = pm.MvNormal('theta', mu=0.5, cov=c0 * sigma * I)
        rating = pm.Normal('rating', mu=0, sd=sigma, observed=user_features)

        for i, (title, movie) in tqdm(enumerate(movies.iterrows())): 
            movies_features = get_movie_features(movies)
            # Expected value of outcome
            mu = user_features.dot(movies_features) * (1 - np.exp(-epoch/s))
            # Likelihood (sampling distribution) of observations
            rating.mu = mu
            
            step = pm.Metropolis()
            trace = pm.sample(1000, step=step, njobs=1, progressbar=False)
            ratings[i] = rating.distribution.random()[0]
    return movies[movies.index == movies.index[ratings.argmax()]]


# In[91]:


#Compute utility U based on user preferences and movie preferences 
def compute_utility(user_features, movie_features, epoch, s):
    res = user_features.dot(movie_features) * (1 - math.exp(-epoch/s))
    return res

#Compute utility U based on user preferences and movie preferences
def compute_novelty(allepoch, s):
    res = []
    for epoch in allepoch:
        res.append(1 - math.exp(-epoch/s))
    return res

def compute_UCB(epoch, Nt):
    return math.sqrt((2 * math.log2(epoch + 1)) / (Nt * epoch))

#selected features from dataframe
def get_movie_features(movie):
    if isinstance(movie, pd.Series):
        return movie[-WANTED_DIM:]
    elif isinstance(movie, pd.DataFrame):
        return get_movie_features(movie.loc[movie.index[0]])
    else:
        raise TypeError("{} should be a Series or DataFrame".format(movie))
        
def iterative_mean(old, new, t):
    return ((t-1) / t) * old + (1/t) * new
    
def update_features(user_features, movie_features, rating, t):
    return iterative_mean(user_features, movie_features * rating, t+1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Content based filtering

# In[92]:


def best_contentbased_recommandation(user_features, movies, epoch, s, UCB=False):
    utilities = np.zeros(movies.shape[0])
    for i, (title, movie) in enumerate(movies.iterrows()):
        movie_features = get_movie_features(movie)
        utilities[i] = compute_utility(user_features, movie_features, epoch - movie.last_t, s)
        if UCB:
            utilities[i] += compute_UCB(epoch, movie.Nt)
    return movies[movies.index == movies.index[utilities.argmax()]]


# ## Collaborative Filtering

# In[93]:


def best_collaborative_recommandation(user_features, user_movies, epoch, s):
    corr = np.zeros(collabo.shape[0])
    corruser = np.zeros(collabo.shape[0])
    # on fait une pearson corr avec tous les autres users -> CLUSTERING
    for collabi, collabrow in enumerate(collabo.iterrows()):
        otheruser_index = collabrow[0]
        otheruser_features = collabrow[1]
        corr[collabi] = float(np.correlate(user_features, otheruser_features)[0])
        corruser[collabi] = otheruser_index
    # on prends les films des 5 plus proche
    idxbestuser = []
    for i in range(10):
        idxmax = corr.argmax()
        idxbestuser.append(corruser[idxmax])
        corruser[idxmax] = 0
    moviesbestuser = users.copy()[users.user.isin(idxbestuser)].index
    # on fait une jointure avec les films du user
    try:
        subsetmovie = user_movies.copy().loc[moviesbestuser]
        subsetmovie = subsetmovie.dropna()
    except:
        print("WARNING : no jointure btw user")
        return best_contentbased_recommandation(user_features, user_movies, epoch, s)
    ## on retourne le mieux coté
    subsetmovie['rating'] = subsetmovie['rating'] * compute_novelty(epoch - subsetmovie.last_t, s)
    maxrating = subsetmovie['rating'].max()
    return subsetmovie[subsetmovie.rating == maxrating].sample()


# ### Results Collaborative vs Content based

# In[94]:


def reinforcement_learning(user, moviestc, algorithm, s, numberSimulation):
    if s<200:
        print("WARNING : s is really small, movies will get often repeated")
    algorithm = algorithm()
    user_features = np.zeros(moviestc.shape[1] - 2)
    movies = moviestc.copy()
    movies = movies[movies.columns.difference(["votes", "rating"])]
    movies.insert(0, 'last_t', np.ones(movies.shape[0]).astype(np.int64))
    movies.insert(0, 't', [i for i in range(movies.shape[0])])
    movies.insert(0, 'rating', user.rating)
    movies.insert(0, 'Nt', np.zeros(movies.shape[0]))
    cumregret = [0]
    accuracy_rmse = [0]
    avg_rating = [0]
    timestamp = []
    for t in range(numberSimulation):
        now = datetime.datetime.now()
        recommandation = algorithm.choice(user_features, movies, t+1, s)
        recommandation_features = get_movie_features(recommandation)
        user_rating = user.loc[recommandation.index[0]].rating
        user_features = algorithm.update_features(user_features, recommandation_features, user_rating, t)
        utility = algorithm.compute_utility(user_features, recommandation_features, t, s)
        cumregret.append(cumregret[-1] + (user_rating - utility ))
        accuracy_rmse.append((user_rating - utility )**2 )
        avg_rating.append(user_rating)
        movies.loc[movies.index.isin(recommandation.index),'last_t'] = t
        movies.loc[movies.index.isin(recommandation.index),'Nt'] += 1
        timestamp.append((datetime.datetime.now() - now).total_seconds())
    return {'cumregret': cumregret, 'accuracy_rmse':accuracy_rmse, 'avg_rating':avg_rating, 'timediff':timestamp}


# In[95]:


def rl_multiple_users(users, movies, algorithms, s=500, N=20, N_USER=50):
    def wrapper_rl_one_user(args):
        return reinforcement_learning(*args)
    results_all = []
    users_sample = users[users.user.isin(pd.Series(users.user.unique()).sample(N_USER))].copy()
    movies_sample = movies[movies.index.isin(users_sample.index.unique())].copy()
    for algo in tqdm(algorithms):
        res_algo = []
        args = []
        for i, name in enumerate(users_sample.user.unique()):
            user = users[users.user == name]
            movies_user = movies_sample[movies_sample.index.isin(user.index)]
            res = reinforcement_learning(user, movies_user, algo, s, N)
            res_algo.append(res)
        results_all.append(res_algo)
    return results_all


# In[96]:


# Keep list consistent
ALGOS      = [partial(LinUCB, 0.5), greedy_choice_no_t_contentbased, random_choice] #, greedy_choice, random_choice]
ALGOS_NAME = ['LinUCB', 'greedy_choice_no_t_contentbased', 'random_choice'] #, 'greedy_choice', 'random_choice']
assert(len(ALGOS) == len(ALGOS_NAME))


# In[104]:


METRICS = ['cumregret', 'accuracy_rmse', 'avg_rating', 'timediff']
TITLE_GRAPH=['Average cumulative regret for each algorithm', 'Average accuracy for each algorithm', 'Average rating for each algorithm', 'Average running time for each algorithm']
X_AXIS = ['Cumulative reget', 'Accuracy (root mean square error)', 'Rating', 'Time']
assert(len(METRICS) == len(TITLE_GRAPH) == len(X_AXIS))


# In[100]:


get_ipython().run_line_magic('time', 'res = rl_multiple_users(users, movies, ALGOS, N=500, N_USER=5, s=200)')


# In[105]:



for metric, tgraph, xaxix in zip(METRICS,TITLE_GRAPH,X_AXIS):
    data = []
    for algo, algon in enumerate(ALGOS_NAME):
        temp = np.average(np.array([i[metric] for i in res[algo]]), axis=0)[1:]
        data.append(go.Scatter(
            x = list([i for i in range(len(temp))]),
            y = temp,
            name=algon
        ))
    layout = dict(title = tgraph,
              xaxis = dict(title = tgraph),
              yaxis = dict(title = xaxix),
    )
    fig = dict(data=data, layout=layout)
    plotly.offline.iplot(fig)


# In[ ]:




