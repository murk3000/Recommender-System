# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:47:34 2022

@author: murk
"""

import sys
import os 

# Allows you to import the file from the first & second exercise as a package
sys.path.append(os.path.abspath('D:\\Grad\\Period 2\\Recommender Systems'))

import simple_recommendations as ex1
import group_recommendations as ex2
import pandas as pd
import numpy as np

def satisfaction(preds, recomm, user, k):
    df = pd.merge(preds[preds['userId']==user].nlargest(k, columns=['rating']), 
                  recomm[recomm['userId']==user], how='inner', on='movieId')
    sum(df['rating_y'])/sum(df.fillna(0)['rating_x'])

def satisfaction2(preds, recomm, user, k):
    sum(recomm[recomm['userId']==user].nlargest(k, columns=['rating'])['rating'])/sum(preds[preds['userId']==user].nlargest(k, columns=['rating'])['rating'])


def seq_recomm(current_recomm, ind_preds, group, k, prev_recomms=[]):
    sat = [satisfaction(ind_preds, current_recomm, u, k) for u in group]
    alpha = max(sat)-min(sat)
    
    df = ind_preds[not ind_preds['movieId'].isin(prev_recomms)]
    df = df[['movieId', 'rating']].groupby('movieId').agg(['mean', 'min']).reset_index().droplevel(0, axis=1)
    df['rating'] = (1-alpha)*df['mean'] + alpha*df['min']
    
    return(df[['movieId', 'rating']])
    

if __name__ == '__main__':
    ratings = pd.read_csv('Ex1/ml-latest-small/ratings.csv')
    
    user_sim = ex1.pearson_user_sim(ratings)
    def u_sim(i,j,user_sim=user_sim):
        try:
            sim = user_sim[(user_sim['user1']==i) & (user_sim['user2']==j)]['sim'].values[0]
        except:
            sim = 0
        finally:
            return sim
    # item_sim = ex1.cosine_item_sim(ratings) # uncomment this if calculating item_sim for the first time
    item_sim = pd.read_pickle('Ex1/item_sim')
    
    u_group= [1,9,13]
    recomm, preds = ex2.groups_rating2(ratings, user_sim, u_group, 
                   ['mean', 'min', 'sum'], 
                   rating_transformation=[lambda x:x, lambda x:x])
    recomm['rating'] = recomm['rating1']

    new_recomm = recomm
    movies = []
    for i in range(4):
        new_recomm = seq_recomm(new_recomm, preds, u_group, 20, movies)
        movies = movies + list(new_recomm['movieId'])
    
    

