import streamlit as st
import pandas as pd
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

# Start Spark session
spark = SparkSession.builder.appName('MovieRecommendation').getOrCreate()

ratings = spark.read.csv('ml-latest-small/ratings.csv', header=True, inferSchema=True)
movies = spark.read.csv('complete_movies.csv', header=True, inferSchema=True)

def gather_top_movies():
    # Compute average ratings and number of ratings per movie
    movie_ratings_stats = ratings.groupBy('movieId').agg(
        F.avg('rating').alias('avg_rating'),
        F.count('userId').alias('num_ratings')
    )

    # Join movies with their ratings statistics
    movies_with_stats = movies.join(movie_ratings_stats, on='movieId')
    popular_movies = movies_with_stats.filter('num_ratings >= 50')

    return popular_movies


def load_movies():
    popular_movies = gather_top_movies()

    movies_copy = movies

    genre_counts = movies_copy.withColumn('genres', F.explode(F.split('genres', '[|]'))).groupBy('genres').count().orderBy(F.desc('count'))

    top_genres = genre_counts.select('genres').rdd.flatMap(lambda x: x).collect()

    # Explode genres into new rows and count each genre
    genre_counts = movies.withColumn('genres', F.explode(F.split('genres', '[|]')))
    genre_counts = genre_counts.groupBy('genres').count().orderBy(F.desc('count'))

    # Collect top genres to the driver (assuming there aren't many genres, otherwise sample or limit)
    top_genres = [row['genres'] for row in genre_counts.collect()]
    selected_genres = random.sample(top_genres, min(3, len(top_genres)))

    # For each selected genre, find the most popular movie
    distinct_movies = []
    for genre in selected_genres:
        genre_movies = popular_movies.filter(popular_movies.genres.contains(genre)) \
                                    .orderBy(F.desc('num_ratings'), F.desc('avg_rating'))
        top_movie = genre_movies.limit(1)
        distinct_movies.append(top_movie)

    # Union all the distinct top movies from each genre into a single DataFrame
    if distinct_movies:
        initial_movies = distinct_movies[0]
        for movie_df in distinct_movies[1:]:
            initial_movies = initial_movies.union(movie_df)

    return initial_movies.toPandas()


def recommend_movies(user_ratings):

    new_user_df = spark.createDataFrame(user_ratings)

    sample_ids = ratings.select('userId').distinct().sample(fraction=0.4)
    # Using a join is better here because it's more efficient than using a filter
    ratings_sampled = ratings.join(sample_ids, on='userId', how='inner')
    ratings_sampled = ratings_sampled.drop('timestamp')
    combined_df = ratings_sampled.union(new_user_df)

    # Convert ratings into a feature vector per user (pivot table style)
    user_item_ratings = combined_df.groupBy("userId").pivot("movieId").agg(F.first("rating"))

    # Fill missing values in user-item matrix, if necessary
    user_item_ratings = user_item_ratings.na.fill(0)

    # Convert to vector column required by ML functions
    assembler = VectorAssembler(inputCols=user_item_ratings.columns[1:], outputCol="features")
    vector_df = assembler.transform(user_item_ratings)

    # Compute the Pearson correlation matrix
    correlation_matrix = Correlation.corr(vector_df, "features", "pearson").head()[0]

    # Extract the last row (new user) to find similarity scores
    similarity_scores = correlation_matrix.toArray()[-1][:-1]

    # Find top similar users based on correlation scores
    top_users = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:30]

    # Fetch their movie ratings
    top_user_ids = [user_item_ratings.select("userId").collect()[i][0] for i in top_users]
    top_users_df = combined_df.filter(combined_df.userId.isin(top_user_ids))

    # Aggregate to find top recommended movies
    movie_recommendations = top_users_df.groupBy("movieId").agg(F.avg("rating").alias("avg_rating"))
    top_movie_ids = movie_recommendations.orderBy(F.desc("avg_rating")).limit(20).select("movieId").rdd.flatMap(lambda x: x).collect()

    # Get movie details
    recommended_movies_with_titles = movies.filter(movies.movieId.isin(top_movie_ids)).select("title", "genres", "Imdb Link", "IMDB Score").toPandas()
    
    return recommended_movies_with_titles

def main():

    st.title("Movie Recommendation System")

    if 'movies' not in st.session_state:
        st.session_state.movies = load_movies()

    if st.button('Refresh Movies'):
        st.session_state.movies = load_movies()

    movies = st.session_state.movies

    ratings_map = {
        'Never Saw It': 0,
        'Bad': 1,
        'Meh': 2,
        'Average': 3,
        'Good': 4,
        'Masterpiece': 5
    }

    if 'ratings' not in st.session_state:
        st.session_state.ratings = {}

    cols = st.columns(3)
    movie_index = 0

    default_image = 'default_image.jpeg'

    for index, row in movies.iterrows():
        with cols[movie_index]:
            #poster_url = row['Poster'] if pd.notna(row['Poster']) else default_image
            st.markdown(
                f"""
                <a href='{row['Imdb Link']}' target='_blank'>
                    <img src='{row['Poster']}' onerror="this.onerror=null;this.src='{default_image}';"><br>
                    {row['title']}
                </a>
                """,
                unsafe_allow_html=True
            )

            st.markdown(

                """
                <style>
                img {
                    height: 150px;
                    width: auto;
                    border-radius: 10px;
                    box-shadow: 2px 2px 10px grey;
                    align-items: center;
                }

                .st-emotion-cache-ocqkz7 {
                    gap: 20px;
                    max-width: 800px;
                    width: 800px;
                }

                @media (max-width: 768px) {
                    .st-emotion-cache-ocqkz7 {
                        gap: 10px;
                        max-width: 400px;
                        width: 400px;
                    }
                }

                .st-emotion-cache-1dx1gwv {
                    padding: 19.33333px 0px 0px;
                }

                .st-emotion-cache-1nxfa6 {
                    gap: 5px;
                }

                .container {
                margin: 10px;
                padding: 10px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                text-align: center;
                }

                </style>
                """,

                unsafe_allow_html=True

            )

            st.session_state.ratings[row['movieId']] = st.select_slider(
                label = f"How do you rate {row['title']}?",
                options = ["Never Saw It", 'Bad', 'Meh', 'Average', 'Good', 'Masterpiece'],
                key = row['movieId']
            )

            st.session_state.ratings[row['movieId']] = ratings_map[st.session_state.ratings[row['movieId']]]

            movie_index += 1

    if st.button('Submit Ratings'):
        new_user_id = ratings.select(F.max('userId')).collect()[0][0] + 1
        user_ratings = [
            {'userId': new_user_id, 'movieId': mid, 'rating': rating}
            for mid, rating in st.session_state.ratings.items() if rating > 0
        ]

        recommended_movies = recommend_movies(user_ratings)

        # if user_ratings:
        #     new_user_df = pd.DataFrame(user_ratings)
        #     if 'user_ratings' not in st.session_state:
        #         st.session_state.user_ratings = new_user_df
        #     else:
        #         st.session_state.user_ratings = pd.concat([st.session_state.user_ratings, new_user_df], ignore_index=True)
            
        st.session_state.user_id = new_user_id + 1
        new_user_id += 1
        st.success("Thanks for submitting your ratings! \n Here are some movies you might like:")
        st.write("If you like the movies, click the download button to save your recommendations.")
        st.dataframe(recommended_movies)
        #st.dataframe(st.session_state.user_ratings)



if __name__ == "__main__":
    main()            