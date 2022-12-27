import sys
import os 

# Allows you to import the file from the first exercise as a package
sys.path.append(os.path.abspath('D:\\Grad\\Period 2\\Recommender Systems'))

import simple_recommendations as ex1
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from itertools import combinations

def borda_count(x):
    return np.argsort(x)

def compute_kendalltau(recomm, ratings_matrix, sim, group):
    overall_tau = []
    for u in group:
        preds = ex1.get_predictions(ratings_matrix, u, sim, ex1.pearson_prediction_2, orig=False, movie_set=recomm['movieId'])
        preds = pd.DataFrame(preds,  columns=['movieId', 'rating'])
        temp = pd.merge(recomm, preds, how='inner', on='movieId') 
        tau, _ = kendalltau(np.argsort(temp['rating']), np.argsort(temp['agg_rating']))
        overall_tau.append(tau)
    
    return overall_tau

def compute_satisfaction(ratings_matrix, u, m_list, sim):
    pass

def groups_rating2(ratings_matrix, sim, group, agg, rating_transformation=[None]):
    ratings = pd.DataFrame([], columns=['userId', 'movieId', 'rating'])
    predictions = pd.DataFrame([], columns=['userId', 'movieId', 'rating'])
    for u in group:
        temp = ex1.get_predictions(ratings_matrix, u, sim, ex1.pearson_prediction_2, orig=False)
        temp = pd.DataFrame(temp,  columns=['movieId', 'rating'])
        temp['userId']=u
        
        predictions = pd.concat([predictions, temp])
        
        # apply a transformation on ratings before caclulating agg; default is no transformation
        for i,f in enumerate(rating_transformation):
            if f is None:
                continue
            temp['rating'+str(i+1)] = f(temp['rating'])
        
        ratings = pd.concat([ratings, temp])
        
    agg_dict = dict([('userId','count')]+[('rating'+str(i+1), agg[i]) for i in range(len(rating_transformation))])
    ratings = ratings.groupby('movieId').agg(agg_dict).reset_index().rename(columns={'userId':'count'})
    ratings = ratings[ratings['count']==len(group)]
    
    
    return ratings.sort_values(by=['rating1'], ascending=False).reset_index(drop=True), predictions

def user_satisfaction(u, preds, group_recomm):
    num = sum(preds[(preds['userId']==u) & (preds['movieId'].isin(group_recomm['movieId']))]['rating'])
    denom = sum(preds[preds['userId']==u].nlargest(len(group_recomm), 'rating')['rating'])
    return num/denom

def user_satisfaction2(u, preds, group_recomm):
    p = preds[preds['userId']==u].nlargest(len(group_recomm), 'rating')
    num = sum(p[(p['userId']==u) & (p['movieId'].isin(group_recomm['movieId']))]['rating'])
    denom = sum(p[p['userId']==u]['rating'])
    return num/denom


def weighted_avg(recomm, group_recomm, preds, group, avg_col, min_col):
    satisfaction = [user_satisfaction(u, preds, group_recomm) for u in group]
    weight = (max(satisfaction) - min(satisfaction))
    recomm['w_avg'] = (1-weight)*recomm[avg_col] + weight*recomm[min_col]
    
    return recomm.sort_values(by=['w_avg'], ascending=False).reset_index(drop=True), weight
    


if __name__ == '__main__':
    # movies = pd.read_csv('ml-latest-small/movies.csv')
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
    recomm, preds = groups_rating2(ratings, user_sim, u_group, 
                   ['mean', 'min', 'sum'], 
                   rating_transformation=[lambda x:x, lambda x:x, borda_count])
    
    # t=pd.merge(user_sim, user_sim, left_on='user1', right_on='user2', how='inner')
    # t[(t['sim_x']+t['sim_y'])/2<-0.6]
    # t[(abs((t['sim_x']+t['sim_y'])/2)<0.2) & (t['user1_x']!=t['user2_x']) & (t['user1_y']!=t['user2_y']) & (t['user2_x']!=t['user1_y'])]
    
    # avg_recomm = groups_rating(ratings, user_sim, u_group, 'mean')    
    # print(avg_recomm.loc[0:20]['movieId'])
    
    # min_recomm = groups_rating(ratings, user_sim, u_group, 'min')    
    # print(min_recomm.loc[0:20]['movieId'])
    
    
    # borda_recomm = groups_rating(ratings, user_sim, [1,2,3], 'sum', rating_transformation=borda_count)
    
    u_group = [609,610,194 ]
    recomm2, preds2 = groups_rating2(ratings, user_sim, u_group, 
                   ['mean', 'min', 'sum'], 
                   rating_transformation=[lambda x:x, lambda x:x, borda_count])
    
    [u_sim(i, j) for i,j in combinations(u_group,2)]    
    
    u_group = [1,9,158]
    recomm3, preds3 = groups_rating2(ratings, user_sim, u_group, 
                   ['mean', 'min', 'sum'], 
                   rating_transformation=[lambda x:x, lambda x:x, borda_count])
    
    n = 20
    weights = []
    r = recomm.copy()
    for i in range(10):
        r, w, = weighted_avg(r, r.loc[0:n], preds, u_group, 'rating1', 'rating2')
        weights.append(w)
    
    # [user_satisfaction2(u, preds2, r.loc[:n]) for u in u_group]
    weights2 = []
    r = recomm2.copy()
    for i in range(10):
        r, w, = weighted_avg(r, r.loc[0:n], preds2, u_group, 'rating1', 'rating2')
        weights2.append(w)
  
    weights3 = []
    r = recomm3.copy()
    for i in range(10):
        r, w, = weighted_avg(r, r.loc[0:n], preds3, u_group, 'rating1', 'rating2')
        weights3.append(w)
    
        
    