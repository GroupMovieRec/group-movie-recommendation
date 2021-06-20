#this file was initially created to get all the posters for all the movies in the Small Movielens Dataset

from bs4 import BeautifulSoup
import imdb
import urllib.request as urllib2
import pandas as pd
import csv

access = imdb.IMDb()

# open the file in the write mode
with open('posters_no_errors.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    row = ['movieId', 'imdbId', 'posterUrl']
    writer.writerow(row)

DATA_POSTERS = "../tud_small_movie_lens_data"
df_links = pd.read_csv(f"{DATA_POSTERS}/links.csv", names=["movieId", "imdbId", "tmdbId"])

count = 0
for movieId, id in zip(df_links['movieId'], df_links['imdbId']):
    id = int(id)
    try:
        movie = access.get_movie(id)
        row = [movieId, id, movie['full-size cover url']]
        with open('posters_no_errors.csv', 'a') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(row)
            f.close()
    except:
        print("error")
        row = [movieId, id, "No URL"]
        with open('posters_no_errors.csv', 'a') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(row)
            f.close()
        continue
    count += 1
    print("Count", count)