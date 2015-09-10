import util
# import cr
from sklearn import cross_validation
import numpy as np
import h5py
import c_rank

f_user_artists = "data/lastfm/user_artists.dat"

f_artists = "data/lastfm/artists.dat"


# Get artists
artists = util.get_artists(f_artists)
a2i = util.convert_to_ind(artists)


# Get users
users = np.array(util.get_users(f_user_artists))
u2i = util.convert_to_ind(users)


# Get user preferences
pref = util.get_user_pref(u2i, a2i, f_user_artists)


# Cross validation sets
user_split = cross_validation.ShuffleSplit(
    len(users), 1, test_size=0.25, random_state=0)


for _, v_ind in user_split:

    # Remove half of the assignments for the validation users
    t_pref = util.remove_half_pref(pref, v_ind)

    # Create a rating matrix for the set
    ratings = util.create_rating_matrix(len(users), len(artists), t_pref)

    # Get the user-based similarity metric
    sim_metric = c_rank.mod_cosine_similarity(ratings)

    # Extract the user-based features
    features = c_rank.user_feature_extract(ratings, sim_metric, 5)

    # Save the similarity metric and features to a file
    ff = h5py.File('lastfm.hdf5', 'w')
    ff.create_dataset('v_ind', data=v_ind)
    ff.create_dataset('sim_metric', data=sim_metric)
    ff.create_dataset('features', data=features)







# # Get the user groups
# train_users = users
# valid_users = users[v_ind]
#
# # Convert the user IDs to indices
# t_u2i = util.convert_to_ind(train_users)
# v_u2i = util.convert_to_ind(valid_users)
#
# # Get the preferences for each of the user groups
# t_pref = util.get_user_pref(t_u2i, a2i, f_user_artists)
# v_pref_targets = util.get_user_pref(v_u2i, a2i, f_user_artists)
#
# # Remove half of the assignments in the validation set
# v_pref = util.remove_half_pref(v_pref_targets)
#
# # Create rating matrices for the sets
# t_ratings = util.create_rating_matrix(len(train_users), len(artists), t_pref)
# v_ratings = util.create_rating_matrix(len(valid_users), len(artists), v_pref)
#
# sim_metric = c_rank.mod_cosine_similarity(t_ratings)
#
# ff = h5py.File('lastfm_sim.hdf5', 'w')
# ff.create_dataset('sim_metric', data=sim_metric)
# ff.create_dataset('train_users', data=train_users)
# ff.create_dataset('valid_users', data=valid_users)
