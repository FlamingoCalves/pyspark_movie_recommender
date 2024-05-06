# Movie Recommendation System (+ Pyspark Implementation)

This repository contains code for a movie recommendation system implemented in two versions: one using Pandas and another using PySpark. This project is designed to demonstrate the application of Python and PySpark in data processing and machine learning for movie recommendations. It combines backend logic with a touch of front-end development using Streamlit, and it is available on Streamlit sharing.

## Repository Structure

- `final_movie_app.py`: The primary application file using Pandas, with integrated HTML and CSS within Streamlit commands for the frontend.
- `final_movie_app_pyspark_version.py`: Implements the same functionality as `final_movie_app.py`, but using PySpark for distributed data processing.
- `requirements.txt`: Specifies Python dependencies required for the application.
- `complete_movies.csv`: Contains metadata about movies.
- `ml-latest-small/`: Includes MovieLens dataset files:
  - `ratings.csv`: User ratings for movies.
  - `movies.csv`: Movie titles and genres.
  - `links.csv`: Links to IMDb and TMDb for each movie.

## Setup

### Prerequisites

Ensure you have Python and Spark (for the PySpark version) installed on your system.

### Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/your-github-username/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
```

## Running the Application
Run the Streamlit application using:

- For the Pandas version:
```bash
streamlit run final_movie_app.py
```

- For the Pyspark version:
```bash
streamlit run final_movie_app_pyspark_version.py
```

Accessing the Application on Streamlit Sharing
The application is also hosted on Streamlit sharing and can be accessed [here.](https://your-streamlit-sharing-link).

## How It Works
The app allows users to rate movies through an interactive interface. It then recommends new movies based on these ratings using a collaborative filtering algorithm, first prototyped with Pandas and then scaled with PySpark.

### Features
- Interactive movie rating system via Streamlit.
- Enhanced user interface with HTML and CSS within Streamlit.
- Demonstrates scalability using PySpark for data processing.
