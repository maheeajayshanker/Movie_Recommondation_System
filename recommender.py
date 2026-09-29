import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, csv_path="data/movies.csv"):
        self.df = pd.read_csv(csv_path)

        self.df["Genre"] = self.df["Genre"].fillna("Unknown")
        self.df["Language"] = self.df["Language"].fillna("Unknown")
        self.df["Rating"] = self.df["Rating"].fillna(self.df["Rating"].median())

        # A movie title can occur more than once in the raw dataset.
        # Keep one representative row per title for recommendations.
        self.movies = (
            self.df.sort_values(
                ["Rating", "Number_of_Ratings"],
                ascending=[False, False]
            )
            .drop_duplicates("Movie_Title")
            .reset_index(drop=True)
        )

        self.genres = sorted(self.movies["Genre"].astype(str).unique())
        self.languages = sorted(self.movies["Language"].astype(str).unique())
        self.movie_titles = self.movies["Movie_Title"].tolist()

        categorical_features = ["Genre", "Language"]
        numeric_features = ["Release_Year", "Rating", "Runtime_Minutes"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("numeric", StandardScaler(), numeric_features),
            ]
        )

        self.feature_matrix = self.preprocessor.fit_transform(
            self.movies[categorical_features + numeric_features]
        )
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

    def recommend(
        self,
        movie_title,
        n=5,
        genre=None,
        language=None,
        min_rating=1.0,
        max_year=None
    ):
        if movie_title not in self.movie_titles:
            return pd.DataFrame()

        movie_index = self.movies.index[
            self.movies["Movie_Title"] == movie_title
        ][0]

        scores = self.similarity_matrix[movie_index].copy()
        candidate = self.movies.copy()
        candidate["Similarity"] = scores

        # Do not recommend the selected movie itself.
        candidate = candidate[candidate["Movie_Title"] != movie_title]

        # Filtering
        if genre is not None:
            candidate = candidate[candidate["Genre"] == genre]
        if language is not None:
            candidate = candidate[candidate["Language"] == language]

        candidate = candidate[candidate["Rating"] >= min_rating]

        if max_year is not None:
            candidate = candidate[candidate["Release_Year"] <= max_year]

        candidate = candidate.sort_values(
            ["Similarity", "Rating", "Number_of_Ratings"],
            ascending=[False, False, False]
        )

        return candidate.head(n)[
            [
                "Movie_Title",
                "Genre",
                "Release_Year",
                "Rating",
                "Number_of_Ratings",
                "Runtime_Minutes",
                "Language",
                "Similarity",
            ]
        ].reset_index(drop=True)


if __name__ == "__main__":
    recommender = MovieRecommender("data/movies.csv")
    print(recommender.recommend("Beyond the Stars", n=5).to_string(index=False))
