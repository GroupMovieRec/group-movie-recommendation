# Caroline Freyer <carolinefreyer@gmail.com>.
# Last modification 19th June 2021.

import rdfnmf_updated
import random
import numpy as np

"""
    Split datasets into test set and training set with a 20-80 split. Trains RDFNMFPlus model for five different 
    groups and compute RMSE for all users and only for users in focus group. 
"""


def run_weighted_nmf():
    # Split for movielens dataset
    movielens_ratings = read_file("training_ratings.csv")

    test_data_indices = random.sample(
        range(len(movielens_ratings)), 2 * len(movielens_ratings) // 10
    )
    test_data = [movielens_ratings[i] for i in test_data_indices]
    training_data = list(set(movielens_ratings) - set(test_data))

    names = [
        "our_ratings/1_all_average.csv",
        "our_ratings/2_all_active.csv",
        "our_ratings/3_all_inactive.csv",
        "our_ratings/4_mix_12active_12inactive.csv",
        "our_ratings/5_mix_3active_1inactive.csv",
    ]
    rmses = {}

    for name in names:
        # Split focus group ratings.
        our_ratings = read_file(name)
        test_data_ours = random.sample(our_ratings, k=2 * len(our_ratings) // 10)
        test_data = test_data + test_data_ours
        our_ratings = list(set(our_ratings) - set(test_data_ours))

        # Extract list of users and movies.
        users_train, users_ours, items = get_sets(training_data, our_ratings)

        # Compute lengths, needed for RDFNMF
        n_train = len(users_train)
        n_ours = len(users_ours)
        m = len(items)

        # Create and train RDFNMFPlus model
        model2 = rdfnmf_updated.RDFNMFPLus(
            n_users_train=n_train, n_items=m, n_users_our=n_ours
        )
        model2.train(ratings_train=training_data, ratings_ours=our_ratings)

        # Predict values for all test set.
        predicted_2 = model2.predict(test_data)

        # Predict values only for our users.
        predicted_ours_2 = model2.predict(test_data_ours)

        # Compute RMSEs.
        rmses[name] = [
            compute_rmse(test_data, predicted_2),
            compute_rmse(test_data_ours, predicted_ours_2),
        ]

    return rmses


# Extract data from files and store in a specific format.
def read_file(filename):
    data = []
    with open(filename) as f:
        lines = f.readlines()[1:]
        for line in lines:
            user_id, movie_id, rating = line.strip().split(",")[:3]
            data.append((user_id, movie_id, float(rating)))
    return data


# Extracts list of users and movies.
def get_sets(train, ours):
    users_train = set()
    users_ours = set()
    items = set()
    for (user, item, r) in train:
        users_train.add(user)
        items.add(item)
    for (user, item, r) in ours:
        users_ours.add(user)
        items.add(item)
    return list(users_train), list(users_ours), list(items)


# Compute RMSE given test set with ground truth and system predictions.
def compute_rmse(X_test, X_pred):
    sse = 0.0
    for i in range(len(X_test)):
        u_test, i_test, r_test = X_test[i]
        u_pred, i_pred, r_pred = X_pred[i]
        assert u_test == u_pred
        assert i_test == i_pred
        sse += (r_test - r_pred) ** 2
    return (sse / len(X_test)) ** 0.5


# Names of files for the five different user groups.
names = [
    "our_ratings/1_all_average.csv",
    "our_ratings/2_all_active.csv",
    "our_ratings/3_all_inactive.csv",
    "our_ratings/4_mix_12active_12inactive.csv",
    "our_ratings/5_mix_3active_1inactive.csv",
]

means = {name: [[] for _ in range(2)] for name in names}

# Run evaluation for 50 different splits of dataset.
for i in range(50):
    rmses = run_weighted_nmf()
    # Record RMSE values
    for j in rmses:
        m2, mo2 = rmses[j]
        means[j][0].append(m2)
        means[j][1].append(mo2)

# Print results.
for name in names:
    print(name)
    print("Means:")
    print(np.mean(means[name][0]), "+/-", np.std(means[name][0]))

    print("Means: Focus on ours")
    print(np.mean(means[name][1]), "+/-", np.std(means[name][1]))
