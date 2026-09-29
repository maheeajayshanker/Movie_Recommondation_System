from recommender import MovieRecommender

r = MovieRecommender("data/movies.csv")

print("Number of unique movies:", len(r.movie_titles))
print("\nSample recommendations:\n")
print(r.recommend(r.movie_titles[0], n=5).to_string(index=False))
