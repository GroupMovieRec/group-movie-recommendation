from flask import Flask, request, render_template, session
import numpy as np
import pandas as pd
from surprise import NMF, Dataset, Reader
from scipy.stats import hmean 
import scipy.sparse as sp
import os
import json
import datetime

from lightfm.data import Dataset as LightFMDataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

import csv
from csv import writer
from csv import reader

import shutil

from weighted_matrix_factorization.weighted_mf import weighted_mf
  


app = Flask(__name__, template_folder='templates')
app.secret_key = "super secret key"

DATA_DIR = "static/data"

# Siamese data
# movies = json.load(open(f'{DATA_DIR}/movies.json'))
# friends = json.load(open(f'{DATA_DIR}/friends.json'))
# ratings = json.load(open(f'{DATA_DIR}/ratings.json'))
# soup_movie_features = sp.load_npz(f'{DATA_DIR}/soup_movie_features_11.npz').toarray()
# df_movies = pd.DataFrame(movies)
# movie_ids = np.array(df_movies.movie_id_ml.unique())
# new_friend_id = len(friends)

#TUD data directory
DATA = "tud_small_movie_lens_data"

#weighted MF data => small Movielens dataset
columns = ["movieId", "title", "genres"]
rdfmf_movies = pd.read_csv(f"{DATA}/movies.csv", names=columns)
rdmf_posters = pd.read_csv(f"tud_poster_data/posters.csv", names=columns)

a = pd.read_csv(f"{DATA}/movies.csv", names=columns)
b = pd.read_csv(f"tud_poster_data/posters.csv", names=columns)
b = b.dropna(axis=1)
merged_a_b = a.merge(b, on='movieId')
merged_a_b = merged_a_b.rename(columns={"movieId": "movieId", "title_x": "title", "genres_x": "genres", "title_y": "imdbId", "genres_y": "posterUrl"})

# dataframe that contains "famous" movies => this tackles the problem of prompting unkown movies to users
# to see how many popular movies are currently chosen for the initial 15 ratings check the number of entries in popular_movies/popular_movie_ids.csv
POPULAR_MOVIES_IDS_PATH = "popular_movies/popular_movie_ids.csv"

def get_popular_movies():
	list_popular_movies_ids = []
	with open(POPULAR_MOVIES_IDS_PATH, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			if row[0] == "movieId":
				continue
			try:
				list_popular_movies_ids.append(row[0])
			except(Exception):
				continue
		popular_movies = merged_a_b[0:0]
		for id in list_popular_movies_ids:
			row = merged_a_b.loc[merged_a_b['movieId'] == id]
			popular_movies = popular_movies.append({'movieId' : row['movieId'].values[0],
                    'title' : row['title'].values[0],
					'genres': row['genres'].values[0],
					'imdbId': row['imdbId'].values[0], 
					'posterUrl': row['posterUrl'].values[0]} , 
                    ignore_index=True)
		return popular_movies

def number_to_sentiment(number):
	if number == 0:
		return "Hungover"
	if number == 1:
		return "Tired"
	if number == 2:
		return "Sad"
	if number == 3:
		return "Happy"
	if number == 4:
		return "Relaxed"
	if number == 5:
		return "Angry"
	if number == 6:
		return "Excited"
	if number == 7:
		return "Fear"
	if number == 8:
		return "Indifferent"
	if number == 9:
		return "Sick"

def number_to_gender(number):
	if number == 0:
		return "Male"
	if number == 1:
		return "Female"

def create_ratings_file(input_file):
	with open(input_file, 'r') as read_obj, \
		open(output_file, 'w', newline='') as write_obj:
		csv_reader = reader(read_obj)
		csv_writer = writer(write_obj)
		for row in csv_reader:
			# Pass the list / row in the transform function to add column text for this row
			transform_row(row, csv_reader.line_num)
			# Write the updated row / list to the output file
			csv_writer.writerow(row)


def add_column_in_csv(input_file, output_file, transform_row):
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
        for row in csv_reader:
            transform_row(row, csv_reader.line_num)
            csv_writer.writerow(row)

def get_scores(reclist, movieIds):
	titles = []
	URLs = []
	scores = []

	count = 1
	for id, score in zip(list(reclist['mIds']), list(reclist['mScores'])):
		titles.append(f"{count}. {merged_a_b.loc[merged_a_b['movieId']==id].title.values[0]}")
		URLs.append(merged_a_b.loc[merged_a_b['movieId']==id].posterUrl.values[0])
		scores.append(score)

		count += 1

	return list(zip(titles, URLs, scores))


def recommendation_mf():
	final_recs = weighted_mf()

	#print("Final recommendations are: ", final_recs)
	mIds = []
	mScores = []
	for rec in final_recs:
		mIds.append(str(rec[0]))
		mScores.append(rec[1])
	reclist = pd.DataFrame({'mIds': mIds, 'mScores': mScores})
	reclist.sort_values(by=['mIds', 'mScores'])
	recommendation = get_scores(reclist, list(merged_a_b[merged_a_b.movieId.isin(mIds)].movieId))

	return recommendation

@app.route('/', methods=['GET', 'POST'])
def main():

	if request.method == 'POST':
		global df_movies

		# Get recommendations!
		if 'run-mf-model' in request.form:
			
			pu = recommendation_mf()

			# Precompute next session's 15 movies to be rated
			session.clear()

			# sample 15 popular movies
			df_popular_movies = get_popular_movies()
			random_merged_a_b = list(df_popular_movies.sample(15).movieId) 

			session['movieIds'] = list(merged_a_b[merged_a_b.movieId.isin(random_merged_a_b)].movieId)
			session['top15'] = list(merged_a_b[merged_a_b.movieId.isin(random_merged_a_b)].title)
			session['top15_posters'] = list(merged_a_b[merged_a_b.movieId.isin(random_merged_a_b)].posterUrl)

			session['counter'] = 0
			session['members'] = 0
			session['userAges'] = []
			session['userGenders'] = []
			session['userFeelings'] = []
			session['arr'] = None

			return(render_template('main.html', settings = {'friendsInfo':False, 'showVote': False, 'people': 0, 'buttonDisable': False,'chooseRecommendation':False, 'recommendation': pu}))


		# Collect friends info
		elif 'person-select-gender-0' in request.form:
			
			# write user credentials to file
			with open('data_tu_delft/user_credentials.csv', 'w') as f:

				# create the csv writer
				writer = csv.writer(f)

				# header
				row = ['userId', 'Gender', 'Feeling', 'Age']

				# write header to csv file
				writer.writerow(row)

				for i in range(session['members']):
					session['userAges'].append(int(request.form.get(f'age-{i}')))
					session['userGenders'].append(int(request.form.get(f'person-select-gender-{i}')))
					feeling = "Feelingless"
					session['userFeelings'].append(feeling)

					#create new row in file
					row = [f'{i}', f"{number_to_gender(session['userGenders'][-1])}", f"{session['userFeelings'][-1]}", f"{session['userAges'][-1]}"]
					writer.writerow(row)
				f.close()

			#create files where data is going to be saved
			with open('data_tu_delft/previous_user_ratings.csv', 'w') as f:
				f.close()
			with open('data_tu_delft/user_ratings.csv', 'w') as f:
				f.close()
			with open('data_tu_delft/ratings.csv', 'w') as f:
				f.close()
			

			return(render_template('main.html', settings = {'friendsInfo':False, 'showVote': True, 'people': session['members'], 'buttonDisable': True,'chooseRecommendation':False, 'recommendation': None}))

		# Choose number of people in the group
		elif 'people-select' in request.form:
			count = int(request.form.get('people-select'))
			session['members'] = count
			session['arr'] = [[0 for x in range(15)] for y in range(count)] 
			return(render_template('main.html', settings = {'friendsInfo':True, 'showVote': False, 'people': count, 'buttonDisable': True,'chooseRecommendation':False, 'recommendation': None}))


		# All people voting
		elif 'person-select-0' in request.form:

			column = []
			# add movie_id header
			column.append(session['movieIds'][session['counter']])

			#open ratings fil
			with open('data_tu_delft/ratings.csv', 'a') as ratings_file:

				# create the csv writer
				writer = csv.writer(ratings_file)

				if session['counter'] == 0:
					# add header
					header_ratings = ['userId', 'movieId', 'rating']
					writer.writerow(header_ratings)
				
				for i in range(session['members']):
					user_rating = int(request.form.get(f'person-select-{i}'))
					session['arr'][i][session['counter']] = user_rating

					#add new row in ratings file
					rating = [i, session['movieIds'][session['counter']], user_rating]
					writer.writerow(rating)

					#add new rating column for user ratings file
					column.append(user_rating)

				ratings_file.close()

			# add new column in csv file
			if session['counter'] == 0:
				shutil.copyfile('data_tu_delft/user_credentials.csv','data_tu_delft/previous_user_ratings.csv')
			else:
				shutil.copyfile('data_tu_delft/user_ratings.csv','data_tu_delft/previous_user_ratings.csv')

			add_column_in_csv('data_tu_delft/previous_user_ratings.csv', 'data_tu_delft/user_ratings.csv', lambda row, line_num: row.append(column[line_num - 1]))

			session['counter'] += 1 
			if session['counter'] < 15:     
				return(render_template('main.html', settings = {'friendsInfo':False, 'showVote': True, 'people': len(request.form), 'buttonDisable': True,'chooseRecommendation':False, 'recommendation': None}))
			else:
				#remove previous user ratings
				os.remove("data_tu_delft/previous_user_ratings.csv")

				# sort ratings file based on user id; use for rdfnmf later
				df = pd.read_csv('data_tu_delft/ratings.csv')
				df = df.sort_values(['userId', 'movieId'], ascending=[True, True])

				#replace numbering 
				small_movie_lens_data=pd.read_csv(f'{DATA}/ratings.csv')
				#find max
				max_user_id=small_movie_lens_data['userId'].max()

				#start numbering based on small-movie-lens-data user ids
				df['userId'] = df['userId'] + max_user_id + 1

				df.to_csv('data_tu_delft/ratings.csv', index=False)

				# no header csv file
				with open("data_tu_delft/ratings.csv",'r') as f:
					with open("data_tu_delft/ratings_no_header.csv",'w') as f1:
						next(f) # skip header line
						for line in f:
							f1.write(line)

				return(render_template('main.html', settings = {'friendsInfo':False, 'showVote': False, 'people': len(request.form), 'buttonDisable': True,'chooseRecommendation':True,  'recommendation': None}))

	elif request.method == 'GET':
		session.clear()
		#top_trending_ids = list(df_movies.sort_values(by="trending_score").head(200).sample(15).movie_id_ml)

		# TUD => 15 random movies from small Movielens dataset
		#random_movies = list(rdfmf_movies.sample(15).movieId)
		
		# sample 15 random movies
		#random_merged_a_b = list(merged_a_b.sample(15).movieId)

		# sample 15 popular movies
		df_popular_movies = get_popular_movies()
		random_merged_a_b = list(df_popular_movies.sample(15).movieId)

		session['counter'] = 0
		session['members'] = 0
		session['userAges'] = []
		session['userGenders'] = []
		session['userFeelings'] = [] #

		session['movieIds'] = list(merged_a_b[merged_a_b.movieId.isin(random_merged_a_b)].movieId)
		session['top15'] = list(merged_a_b[merged_a_b.movieId.isin(random_merged_a_b)].title)
		session['top15_posters'] = list(merged_a_b[merged_a_b.movieId.isin(random_merged_a_b)].posterUrl)

		session['arr'] = None

		return(render_template('main.html', settings = {'showVote': False, 'people': 0, 'buttonDisable': False, 'recommendation': None}))

@app.route('/static/<path:path>')
def serve_dist(path):
	return send_from_directory('static', path)

if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	host= '0.0.0.0'
	app.run(host=host, port=port)
	
	#app.run(host="127.0.0.1", port=8888, debug=True)
	#app.run()
