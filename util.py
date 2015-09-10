import copy
import numpy as np


def get_artists(f_name):
    artists = []
    with open(f_name, 'r') as f:
        next(f)     # skip the header
        for line in f:
            artist_info = line.strip().split('\t')
            artists.append(artist_info[0])
    return artists


def convert_to_ind(items):
    item2index = dict()
    for i, item in enumerate(items):
        item2index[item] = i
    return item2index


def get_users(f_name):
    users = []
    with open(f_name, 'r') as f:
        next(f)
        for line in f:
            user, _, _ = line.strip().split('\t')
            if user not in users:
                users.append(user)
    return users


def get_user_pref(u2i, a2i, f_name):
    pref = dict()
    with open(f_name, 'r') as f:
        next(f)
        for line in f:
            user, artist, _ = line.strip().split('\t')
            artist_id = a2i[artist]
            if user in u2i:
                user_id = u2i[user]
                if user_id in pref:
                    pref[user_id].add(artist_id)
                else:
                    pref[user_id] = set([artist_id])
    return pref


def remove_half_pref(pref, users):
    new_pref = copy.deepcopy(pref)
    for user in users:
        num_prefs = len(new_pref[user])
        for _ in xrange(num_prefs / 2):
            new_pref[user].pop()
    return new_pref


def create_rating_matrix(num_users, num_items, pref):
    ratings = np.zeros(shape=(num_users, num_items))
    for user in pref.keys():
        for item in pref[user]:
            ratings[user, item] = 1
    return ratings
