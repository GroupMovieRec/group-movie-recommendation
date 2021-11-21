# this file was initially created to match the popular movies from top_1000_IMDB_movies.csv with the movies in tud_small_movie_lens_data/movies.csv
# by doing this we were able to initially display 15 popular movies to users, thus, increasing the chances of getting informative rating from users
import imdb
import csv
from operator import itemgetter
from difflib import SequenceMatcher
   
ia = imdb.IMDb()
ours = []
SMALL_MOVIELENS_PATH = "../tud_small_movie_lens_data/movies.csv"
top_movies = []

with open('top_1000_IMDB_movies.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[2] == 'year':
            continue
        elif int(row[2]) >= 2019:
            continue
        else: top_movies.append([row[1].lower(), row[2]])

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

movie_titles = []
with open(SMALL_MOVIELENS_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1] == "title":
            continue
        try:
            movie_titles.append([row[1].split(" (")[0].lower().replace(",", ""), int(row[1][-5:].split(")")[0]), row[0]])
        except(Exception):
            continue
print("Number of movies in dataset: ", len(movie_titles))

# order movie titles by their ids
movie_titles = sorted(movie_titles, key=itemgetter(2)) 

with open('popular_movie_ids.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    row = ['movieId']
    # write a row to the csv file
    writer.writerow(row)

count = 0
numOfFamousMovies = 0
for movie in top_movies:
    count += 1
    print("Top movie:", movie)
    for title in movie_titles:
        similarity = similar(movie[0], title[0])
        if similarity > 0.7 and int(movie[1]) == int(title[1]):
            print("Count:", count, "Match", similarity, movie, title, "Id: ", title[2])
            numOfFamousMovies += 1
            row = [title[2]]
            with open('popular_movie_ids.csv', 'a') as f:
                # create the csv writer
                writer = csv.writer(f)
                # write a row to the csv file
                writer.writerow(row)
                f.close()
print(numOfFamousMovies)
list_popular_movies = []