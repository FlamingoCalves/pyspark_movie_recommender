import streamlit as st
import pandas as pd
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('complete_movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

def gather_top_movies():
    movie_ratings_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('userId', 'size')
    )

    movies_with_stats = pd.merge(movies, movie_ratings_stats, on='movieId')

    popular_movies = movies_with_stats[movies_with_stats['num_ratings'] >= 50]
    return popular_movies


def load_movies():
    popular_movies = gather_top_movies()
    movies_copy = movies.copy()

    movies_copy['genres'] = movies_copy['genres'].str.split('|')
    movies_copy = movies_copy.explode('genres')

    genre_counts = movies_copy.groupby('genres').size().reset_index(name='count')

    top_genres = genre_counts[genre_counts['count'] > 50]['genres'].tolist()
    selected_genres = random.sample(top_genres, min(3, len(top_genres)))

    diverse_movies = []
    selected_movies = set()

    for genre in selected_genres:
        movie = popular_movies[popular_movies['genres'].str.contains(genre)] \
                                .sort_values(['num_ratings', 'avg_rating'], ascending=False) #Find all the movies that contain the selected genre
        
        movie = movie[~movie['title'].isin(selected_movies)] #Remove the movies that have already been selected
        
        if not movie.empty: #If there's a movie in the 'movie' list
            selected_movie = random.choice(movie.to_dict(orient='records')) #Then pick a random movie from the 'movie' list
            diverse_movies.append(selected_movie) #Add the selected movie to the 'diverse_movies' list
            selected_movies.add(selected_movie['title']) #Add the selected movie to the 'selected_movies' set so that it's not selected again

        else:
            pass

    initial_movies = pd.DataFrame(diverse_movies)
    return initial_movies

def recommend_movies(user_ratings):
    # Generate a new user ID that does not clash with existing user IDs
    #new_user_id = ratings['userId'].max() + 1
    new_user_df = pd.DataFrame(user_ratings, columns=['userId', 'movieId', 'rating'])
    new_user_id = new_user_df['userId'].iloc[0]

    new_user_df_length = len(new_user_df) * -1
    
    # Sample approximately 40% of the user IDs
    sample_ids = ratings['userId'].drop_duplicates().sample(frac=0.4)
    
    # Filter the ratings DataFrame for sampled user IDs
    ratings_sampled = ratings[ratings['userId'].isin(sample_ids)]
    
    # Combine and prepare the DataFrame
    combined_df = pd.concat([ratings_sampled, new_user_df], ignore_index=True)
    
    if combined_df['rating'].isnull().any():
        combined_df['rating'].fillna(combined_df['rating'].mean(), inplace=True)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(combined_df[['rating']]) 
    
    # Calculate mean similarity score for the new user's rows (excluding their own similarity scores)
    mean_similarity_scores = similarity_matrix[new_user_df_length:, :new_user_df_length].mean(axis=0)
    
    # Create a DataFrame for the mean similarity scores
    final_similarity_df = pd.DataFrame(mean_similarity_scores, columns=['mean_similarity'])
    combined_df = combined_df.join(final_similarity_df)
    
    # Find the top similar users (excluding the new user)
    top_similar_users = combined_df[combined_df['userId'] != new_user_id]
    top_similar_user_ids = top_similar_users.groupby('userId')['mean_similarity'].mean().nlargest(30).index
    
    # Select the ratings from the top similar users
    top_similar_users_ratings = top_similar_users[top_similar_users['userId'].isin(top_similar_user_ids)]
    
    # Get the top recommended movies based on these users' ratings
    rec_movies = top_similar_users_ratings.groupby('movieId')['rating'].mean().nlargest(20)
    
    # Final recommended movies with titles
    recommended_movies_with_titles = movies[movies['movieId'].isin(rec_movies.index)][['title', 'genres', 'Imdb Link', 'IMDB Score']]
    
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
        new_user_id = ratings['userId'].max() + 1
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