import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt


def pearson_user_sim(ratings):
    all_watched = pd.merge(ratings, ratings, how='inner', on='movieId')
    user_features = all_watched[['userId_x', 'userId_y', 'rating_x', 'rating_y']].groupby(['userId_x', 'userId_y']).agg(['mean', 'count']).reset_index()
    user_features.columns = ['userId_x', 'userId_y', 'rx_mean', 'rx_count', 'ry_mean', 'ry_count']

    all_watched = pd.merge(all_watched, user_features, how='inner', on=['userId_x', 'userId_y'])
    all_watched = all_watched[all_watched['rx_count']>=3]
    all_watched['rating_x'] = all_watched['rating_x'] - all_watched['rx_mean']
    all_watched['rating_y'] = all_watched['rating_y'] - all_watched['ry_mean']
    
    all_watched['rating_xy'] = all_watched['rating_x']*all_watched['rating_y']
    all_watched['rating_x2'] = all_watched['rating_x']**2
    all_watched['rating_y2'] = all_watched['rating_y']**2
    user_features = all_watched[['rating_x', 'rating_y', 'rating_xy', 'rating_x2', 'rating_y2', 'userId_x', 'userId_y']].groupby(['userId_x', 'userId_y']).agg(['sum']).reset_index()
    user_features.columns = ['userId_x', 'userId_y', 'rx_sum', 'ry_sum', 'rxy_sum', 'rx2_sum', 'ry2_sum']
    
    
    user_sim = user_features[['userId_x', 'userId_y', 'rxy_sum']]
    user_sim['denom'] = np.sqrt(user_features['rx2_sum'])*np.sqrt(user_features['ry2_sum'])
    
    user_sim['sim'] = user_sim['rxy_sum']/user_sim['denom']
    user_sim['user1'] = user_sim['userId_x']
    user_sim['user2'] = user_sim['userId_y']
    
    return user_sim[['user1', 'user2', 'sim']].fillna(0)

def pearson_prediction_2(ratings, user1, movie, user_sim):
    """
    Returns the predicted rating of a movie for a user
    """
    sim = user_sim[(user_sim['user1']==user1)][['user2', 'sim']]
    sim= pd.merge(sim, ratings[ratings['movieId']==movie][['userId', 'rating']], how='inner', left_on='user2', right_on='userId')
    
    means = ratings[['userId', 'rating']].groupby('userId').agg('mean').reset_index().rename(columns={'rating':'mean_rating'})
    sim = pd.merge(sim, means, how='inner', on='userId').fillna(0)
    
    sim['rating']=sim['rating']-sim['mean_rating']
    
    denom = sum(abs(sim['sim']))
    if denom == 0:
        return 0
    else:
        return means[means['userId']==user1]['mean_rating'].values[0] + sum(sim['sim']*(sim['rating']))/denom
        
def cosine_item_sim(ratings):
    all_users = pd.merge(ratings, ratings, how='inner', on='userId')
    user_features = ratings[['userId', 'rating']].groupby(['userId']).agg('mean').reset_index()
    user_features.columns = ['userId', 'r_mean']

    all_users = pd.merge(all_users, user_features, how='inner', on=['userId'])
    
    del user_features
    
    item_features = all_users[['movieId_x', 'movieId_y', 'rating_x']].groupby(['movieId_x', 'movieId_y']).agg('count').reset_index()
    item_features.columns = ['movieId_x', 'movieId_y', 'ry_count']
    all_users = pd.merge(all_users, item_features, how='inner', on=['movieId_x', 'movieId_y'])

    del item_features
    
    all_users = all_users[all_users['ry_count']>=2]
    all_users['rating_x'] = all_users['rating_x'] - all_users['r_mean']
    all_users['rating_y'] = all_users['rating_y'] - all_users['r_mean']
    
    all_users['rating_xy'] = all_users['rating_x']*all_users['rating_y']
    all_users['rating_x2'] = all_users['rating_x']**2
    all_users['rating_y2'] = all_users['rating_y']**2
    
    
    all_users = all_users[['rating_x', 'rating_y', 'rating_xy', 'rating_x2', 'rating_y2', 'movieId_x', 'movieId_y']]
    item_features = all_users.groupby(['movieId_x', 'movieId_y']).agg(['sum']).reset_index()
    
    del all_users
    
    item_features.columns = ['movieId_x', 'movieId_y', 'rx_sum', 'ry_sum', 'rxy_sum', 'rx2_sum', 'ry2_sum']
    
    item_sim = item_features[['movieId_x', 'movieId_y', 'rxy_sum']]
    item_sim['denom'] = np.sqrt(item_features['rx2_sum'])*np.sqrt(item_features['ry2_sum'])
    
    item_sim['sim'] = item_sim['rxy_sum']/item_sim['denom']
    item_sim['item1'] = item_sim['movieId_x']
    item_sim['item2'] = item_sim['movieId_y']
    
    return item_sim[['item1', 'item2', 'sim']].fillna(0)

def cosine_prediction(ratings, user, item1, item_sim):
    """
    Returns the predicted rating of a movie for a user 
    """
    sim = item_sim[(item_sim['item1']==item1)][['item2', 'sim']]
    sim = pd.merge(sim, ratings[ratings['userId']==user][['movieId', 'rating']], how='inner', left_on='item2', right_on='movieId')
    
    denom = sum(abs(sim['sim']))
    if denom == 0:
        return 0
    else:
        return sum(sim['sim']*(sim['rating']))/denom
    
def get_predictions(ratings, user, sim, pred, orig=False, movie_set = None):
    user_movies = ratings[ratings['userId']==user]['movieId'].unique()
    if movie_set is None:
        movie_set = ratings['movieId'].unique()
    movie_recomm = []
    for j in movie_set:
        if j in user_movies:
            if orig:
                movie_recomm.append((j, ratings[(ratings['userId']==user) & (ratings['movieId']==j)]['rating']))
            continue
        movie_recomm.append((j, pred(ratings, user, j, sim)))
    movie_recomm.sort(key=lambda y:-y[1])
    return movie_recomm
        
    
if __name__ == '__main__':
    # links = pd.read_csv('ml-latest-small/links.csv')
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    # tags = pd.read_csv('ml-latest-small/tags.csv')
    
    print(ratings.head())
    print(ratings.describe())
    
    
    user_sim = pearson_user_sim(ratings)
    userid = 1
    print(f'Top 20 Users similar to User {userid}')
    print(user_sim[user_sim['user1']==userid][['user2', 'sim']].sort_values(by=['sim'], ascending=False).head(20))
            
    movie_recomm = get_predictions(ratings, userid, user_sim, pearson_prediction_2, orig=False)
    print(f'Top 20 Movies to Recommend to User {userid}')
    print(pd.DataFrame(movie_recomm[1:20], columns=['movieId', 'rating']))
    
    # item_sim = cosine_item_sim(ratings)
    
    item_sim = pd.read_pickle('item_sim')
    movieid = 1
    print(f'Top 20 Movies similar to Movie {movieid} {movies[movies["movieId"]==movieid]}')
    print(pd.merge(movies, item_sim[item_sim['item1']==movieid][['item2', 'sim']].sort_values(by=['sim'], ascending=False).head(21), how='inner', left_on='movieId', right_on='item2'))

    movie_recomm2 = get_predictions(ratings, userid, item_sim, cosine_prediction, orig=False)
    print(f'Top 20 Movies to Recommend to User {userid}')
    print(pd.DataFrame(movie_recomm2[1:20], columns=['movieId', 'rating']))
    

# all_common_lengths = []
# for ind, i in enumerate(set(ratings['userId'])):
#     print(i)
#     for j in list(set(ratings['userId']))[ind:]:
#         all_common_lengths.append(len(common_movies(ratings, i, j)))
# val, freq = np.unique(all_common_lengths, return_counts=True)
# plt.plot(val[:10], freq[:10]) 
# sum(freq[1:])/sum(freq)
# sum(freq[2:])/sum(freq)
# sum(freq[3:])/sum(freq)
# sum(freq[4:])/sum(freq)
# sum(freq[5:])/sum(freq)
# # with a condition of >=3 common movies condition we lose 30% of interactions
# # seems to be a good compromise between using all data and data reliability
 
# all_common_lengths = []
# for ind, i in enumerate(set(ratings['movieId'])):
#     print(i)
#     for j in list(set(ratings['movieId']))[ind:]:
#         all_common_lengths.append(len(common_users(ratings, i, j)))
# val2, freq2 = np.unique(all_common_lengths, return_counts=True)
# plt.plot(val2[:10], freq2[:10]) 
# sum(freq2[1:])/sum(freq2)
# sum(freq2[2:])/sum(freq2)
# sum(freq2[3:])/sum(freq2)
# sum(freq2[4:])/sum(freq2)
# sum(freq2[5:])/sum(freq2)
# # with a condition >= 2 we lose 90% of the data (65% if you filter for >= 1 common users)
# # would have to work with this if we want to use as much data as possible