# Caroline Freyer <carolinefreyer@gmail.com>.
# Last modification 19th June 2021.

import rdfnmf
import rdfnmf_updated
import random
import nmf
import numpy as np

'''
    Split datasets into test set and training set with a 20-80 split. Trains three different models 
    NMF, RDFNMF, RDFNMFPlus and computes RMSE for all users, only for users in focus group, bottom 10%
    of users, and bottom 10% of items in the test set. 
'''


def run_weighted_nmf():
    # Read files: Use focus group with all users having average activity.
    our_ratings = read_file("our_ratings/1_all_average.csv")
    movielens_ratings = read_file("training_ratings.csv")

    # Split for movielens dataset
    test_data_indices = random.sample(range(len(movielens_ratings)), 2 * len(movielens_ratings) // 10)
    test_data = [movielens_ratings[i] for i in test_data_indices]
    training_data = list(set(movielens_ratings) - set(test_data))

    # Split focus group ratings.
    test_data_ours = random.sample(our_ratings, k=2 * len(our_ratings) // 10)
    test_data = test_data + test_data_ours
    our_ratings = list(set(our_ratings) - set(test_data_ours))

    # Extract bottom 10% of users and items from test data.
    test_data_user_long_tail, test_data_item_long_tail = get_10_percentile_test_data(movielens_ratings + our_ratings,
                                                                                     test_data)

    # Extract list of users and movies.
    users_train, users_ours, items = get_sets(training_data, our_ratings)

    # Compute lengths, needed for RDFNMF
    n_train = len(users_train)
    n_ours = len(users_ours)
    m = len(items)

    # Combine training set for NMF and RDFNMF models.
    combined = training_data + our_ratings

    # Create and train models
    model0 = nmf.NMF(n_users=n_train + n_ours, n_items=m, n_epochs=20)
    model1 = rdfnmf.RDFNMF(n_users=n_train + n_ours, n_items=m, n_epochs=20)
    model2 = rdfnmf_updated.RDFNMFPlus(n_users_train=n_train, n_items=m, n_users_our=n_ours, n_epochs=20)

    model0.train(combined)
    model1.train(combined)
    model2.train(ratings_train=training_data, ratings_ours=our_ratings)

    # Predict values for all test set.
    predicted_0 = model0.predict(test_data)
    predicted_1 = model1.predict(test_data)
    predicted_2 = model2.predict(test_data)

    # Predict values only for our users.
    predicted_ours_0 = model0.predict(test_data_ours)
    predicted_ours_1 = model1.predict(test_data_ours)
    predicted_ours_2 = model2.predict(test_data_ours)

    # Predict values for only long-tail users.
    predicted_ltu_0 = model0.predict(test_data_user_long_tail)
    predicted_ltu_1 = model1.predict(test_data_user_long_tail)
    predicted_ltu_2 = model2.predict(test_data_user_long_tail)

    # Predict values for only long-tail items.
    predicted_lti_0 = model0.predict(test_data_item_long_tail)
    predicted_lti_1 = model1.predict(test_data_item_long_tail)
    predicted_lti_2 = model2.predict(test_data_item_long_tail)

    # Compute RSMEs.
    rmses = [compute_rmse(test_data, predicted_0), compute_rmse(test_data, predicted_1),
             compute_rmse(test_data, predicted_2), compute_rmse(test_data_ours, predicted_ours_0),
             compute_rmse(test_data_ours, predicted_ours_1), compute_rmse(test_data_ours, predicted_ours_2),
             compute_rmse(test_data_user_long_tail, predicted_ltu_0),
             compute_rmse(test_data_user_long_tail, predicted_ltu_1),
             compute_rmse(test_data_user_long_tail, predicted_ltu_2),
             compute_rmse(test_data_item_long_tail, predicted_lti_0),
             compute_rmse(test_data_item_long_tail, predicted_lti_1),
             compute_rmse(test_data_item_long_tail, predicted_lti_2)]

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
    sse = 0.
    for i in range(len(X_test)):
        u_test, i_test, r_test = X_test[i]
        u_pred, i_pred, r_pred = X_pred[i]
        assert (u_test == u_pred)
        assert (i_test == i_pred)
        sse += (r_test - r_pred) ** 2
    return (sse / len(X_test)) ** .5


# Get bottom 10% of users and items from test data.
def get_10_percentile_test_data(dataset, test_data):
    ratings_per_user = {}
    ratings_per_item = {}
    for (u, i, r) in dataset:
        if u in ratings_per_user:
            ratings_per_user[u] += 1
        else:
            ratings_per_user[u] = 1

        if i in ratings_per_item:
            ratings_per_item[i] += 1
        else:
            ratings_per_item[i] = 1
    limit_user = len(ratings_per_user) // 10
    ratings_per_user_sorted = dict(sorted(ratings_per_user.items(), key=lambda item: item[1]))
    users_10_percent = list(ratings_per_user_sorted.keys())[:limit_user]
    limit_item = len(ratings_per_item) // 10
    ratings_per_item_sorted = dict(sorted(ratings_per_item.items(), key=lambda item: item[1]))
    item_10_percent = list(ratings_per_item_sorted.keys())[:limit_item]

    test_set_user_long_tail = []
    test_set_item_long_tail = []

    for (u, i, r) in test_data:
        if u in users_10_percent:
            test_set_user_long_tail.append((u, i, r))

        if i in item_10_percent:
            test_set_item_long_tail.append((u, i, r))

    return test_set_user_long_tail, test_set_item_long_tail


means = [[] for _ in range(12)]

# Run evaluation for 50 different splits of dataset.
for i in range(50):
    results = run_weighted_nmf()
    for i in range(12):
        means[i].append(results[i])

# Print results.
print("Means: NMF, RDF, RDF++ Comparison")
print(np.mean(means[2]), "+/-", np.std(means[2]))
print(np.mean(means[1]), "+/-", np.std(means[1]))
print(np.mean(means[0]), "+/-", np.std(means[0]))

print("Means: NMF, RDF, RDF++ Comparison: Focus on ours")
print(np.mean(means[5]), "+/-", np.std(means[5]))
print(np.mean(means[4]), "+/-", np.std(means[4]))
print(np.mean(means[3]), "+/-", np.std(means[3]))

print("Means: NMF, RDF, RDF++ Comparison: Focus on 10% long-tail users")
print(np.mean(means[8]), "+/-", np.std(means[8]))
print(np.mean(means[7]), "+/-", np.std(means[7]))
print(np.mean(means[6]), "+/-", np.std(means[6]))

print("Means: NMF, RDF, RDF++ Comparison: Focus on 10% long-tail items")
print(np.mean(means[11]), "+/-", np.std(means[11]))
print(np.mean(means[10]), "+/-", np.std(means[10]))
print(np.mean(means[9]), "+/-", np.std(means[9]))
