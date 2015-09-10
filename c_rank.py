import numpy as np


FEATURE_AVG = 0     # Mean closeness
FEATURE_STD = 1     # Standard deviation of closeness
FEATURE_MAX = 2     # Max closeness
FEATURE_MIN = 3     # Min closeness
FEATURE_CON = 4     # Confidence


def user_feature_extract(ratings, sim_metric, k=5):
    ''' Feature Extraction: User-Based Approach. '''
    num_users, num_items = ratings.shape
    features = np.zeros(shape=(num_users, num_items, 5))
    for user in xrange(num_users):
        print user, "/", num_users
        for item in xrange(num_items):
            TIE_nm = k_closest_user_sim(user, item, sim_metric, ratings, k)
            _set_features(TIE_nm, features[user, item], k)
    return features


def user_sub_feature_extract(users, ratings, sim_metric, k=5):
    _, num_items = ratings.shape
    num_users = len(users)
    features = np.zeros(shape=(num_users, num_items, 5))
    for user in users:
        print user, "/", num_users
        for item in xrange(num_items):
            TIE_nm = k_closest_user_sim(user, item, sim_metric, ratings, k)
            _set_features(TIE_nm, features[user, item], k)
    return features


def item_feature_extract(ratings, sim_metric, k=5):
    ''' Feature Extraction: User-Based Approach.
    Summarize preferences expressed by u for the items that are similar to v.
    It is adventageous to use the item-based neighbourhood approach when the
    number of items is smaller than the number of users.
    '''
    num_users, num_items = ratings.shape
    features = np.zeros(shape=(num_users, num_items, 5))
    for user in xrange(num_users):
        print user, "/", num_users
        for item in xrange(num_items):
            TIE_nm = k_closest_items_sim(user, item, sim_metric, ratings, k)
            _set_features(TIE_nm, features[user, item], k)
    return features


def _set_features(closeness, features, k):
    if len(closeness) > 0:
        features[FEATURE_AVG] = np.mean(closeness)
        features[FEATURE_STD] = np.std(closeness)
        features[FEATURE_MAX] = np.max(closeness)
        features[FEATURE_MIN] = np.min(closeness)
        features[FEATURE_CON] = len(closeness) / float(k)
    return features


def k_closest_items_sim(user, item, sim_metric, ratings, k):
    '''
    Get the cosine similarities of the 'k' most similar items that were rated
    by 'user' and are close to the specified 'item'.
    '''
    item_range = np.nonzero(ratings[user, :])[0]
    item_range = item_range[item_range != item]
    sim_to_item = sim_metric[item, :][item_range]
    return sorted(sim_to_item, reverse=True)[:k]


def k_closest_user_sim(user, item, sim_metric, ratings, k):
    '''
    Get the cosine similarities of the 'k' most similar users to 'user'
    that also rated the specified 'item'.
    '''
    user_range = np.nonzero(ratings[:, item])[0]
    user_range = user_range[user_range != user]     # Don't include themselves
    sim_to_user = sim_metric[user, :][user_range]
    return sorted(sim_to_user, reverse=True)[:k]


def normalize_user_ratings(ratings):
    ''' Rescale the rows of R(u, :) of the binary matrix to have unit form
    for every user. '''
    norm_ratings = ratings.T / ratings.sum(axis=1)
    return norm_ratings.T


def rescale_matrix(matrix):
    ''' Rescale matrix to have unit form for every row. '''
    matrix /= np.sqrt((matrix ** 2).sum(axis=1))[:, None]
    return matrix


def mod_cosine_similarity(ratings):
    '''
    Cosine distance with two modifications from:
        Karypis, G. (2001, October). Evaluation of item-based top-n
        recommendation algorithms. In Proceedings of the tenth international
        conference on Information and knowledge management (pp. 247-254). ACM.

    These modifications account for popularity and user biases, which were
    found to significantly improve performance.

    The modifications include:
    1.  The rows R(u,:) of the binary matrix are rescaled to have unit norm
        for every user u.
    2.  The cosine distance between items computed by using the rescaled
        matrix was also reached to have unit norm for every item.
    '''
    # Modification 1:
    ratings = normalize_user_ratings(ratings)

    # Get similarities
    user_sim = user_similarity(ratings)
    # item_sim = item_similarity(ratings)

    # Modification 2:
    user_sim = rescale_matrix(user_sim)
    # item_sim = rescale_matrix(item_sim)

    return user_sim


def user_similarity(ratings):
    return _similarity_matrix(ratings)


def item_similarity(ratings):
    return _similarity_matrix(ratings.T)


def _similarity_matrix(ratings):
    num_items, _ = ratings.shape
    # sim = np.zeros(shape=(num_items, num_items))
    sim = np.identity(num_items)
    # for item1 in xrange(num_items):
    for item1 in xrange(num_items - 1):
        # for item2 in range(item1, num_items):
        for item2 in range(item1 + 1, num_items):
            # Cosine similarity is symmetric
            sim[item1, item2] = _cosine_similarity(ratings, item1, item2)
            sim[item2, item1] = sim[item1, item2]
            print item1, item2, sim[item1, item2]
    return sim


def _cosine_similarity(ratings, item1, item2):
    '''
    Compute the cosine similarity of item1 and item2.

    Note: The ratings of item1 and item2 must be in the form:
        ratings[item1, :] and ratings[item2, :], respectively.

    Returns:
    --------
    The resulting similarity ranges from -1 meaning exactly opposite, to 1
    meaning exactly the same, with 0 indicating orthogonality (decorrelation),
    and in-between values indicating intermediate similarity or dissimilarity.
    '''
    similarity = ratings[item1].T.dot(ratings[item2])
    item1_mag = np.sqrt((ratings[item1] ** 2).sum())
    item2_mag = np.sqrt((ratings[item2] ** 2).sum())
    similarity /= (item1_mag * item2_mag)
    return similarity
