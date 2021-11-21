import rdfnmf_updated
import numpy as np

def run_weighted_nmf():
    #Read in two data sets: training set and ratings from interface.
    our_ratings = read_file("our_ratings.csv")
    movielens_ratings = read_file("training_ratings.csv")

    #Extract list of users and movies.
    users_train, users_ours, items = get_sets(movielens_ratings, our_ratings)

    #Compute lengths, needed for RDFNMF.
    n_train = len(users_train)
    n_ours = len(users_ours)
    m = len(items)

    #Create and train model.
    model= rdfnmf_updated.RDFNMFPlus(n_users_train=n_train, n_items = m, n_users_our=n_ours)
    model.train(ratings_train=movielens_ratings, ratings_ours=our_ratings)

    #Construct test data: all movie and our users combinations.
    test_data = gen_test_data(users_ours, items)

    #Predict scores for all movie and user combinationrs.
    predicted = model.predict(test_data)

    #Extract top ten movies.
    top_ten = get_top_ten(predicted,items, n_ours)
    print(top_ten)


#Method to read ratings from csv file in the required format.
#Ratings file must be of the form user_id, movie_id, rating per row and have a header.
def read_file(filename):
    data = []
    with open(filename) as f:
        #drop header
        lines = f.readlines()[1:]
        for line in lines:
            user_id, movie_id, rating = line.strip().split(",")[:3]
            data.append((user_id, movie_id, float(rating)))
    return data

#Extracts list of users and movies.
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

#Creates a test set for all combinations of our users and movies.
def gen_test_data(users_ours, items):
    test_data = []
    for i in items:
        for u in users_ours:
            test_data.append((u,i,float(0)))
    return test_data

#Extract top ten movies with highest mean score across our users.
def get_top_ten(predictions, items, n):
        l = int(len(predictions)/n)
        scores = [s for (u,i,s) in predictions]
        predictions_split = np.array_split(scores,l)
        mean_prediction = [(items[i],np.mean(predictions_split[i])) for i in range(l)]
        top_ten = sorted(mean_prediction, key = lambda x: x[1])[-10:]
        top_ten = top_ten[::-1]
        return top_ten

run_weighted_nmf()
