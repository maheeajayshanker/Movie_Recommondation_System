import streamlit as st
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

@st.cache_resource
def load_recommender():
    return MovieRecommender("data/movies.csv")

recommender = load_recommender()

st.title("🎬 Movie Recommendation System")
st.write("A content-based movie recommendation system using filtering, feature preprocessing, and cosine similarity.")

with st.sidebar:
    st.header("🔎 Recommendation Filters")
    genre_options = ["All"] + recommender.genres
    language_options = ["All"] + recommender.languages

    genre = st.selectbox("Genre", genre_options)
    language = st.selectbox("Language", language_options)
    min_rating = st.slider("Minimum Rating", 1.0, 10.0, 1.0, 0.1)
    max_year = st.slider(
        "Release Year",
        int(recommender.df["Release_Year"].min()),
        int(recommender.df["Release_Year"].max()),
        int(recommender.df["Release_Year"].max())
    )
    n = st.slider("Number of Recommendations", 1, 10, 5)

st.subheader("🎞️ Select a Movie")
movie_titles = recommender.movie_titles
selected_movie = st.selectbox("Movie", movie_titles)

if st.button("✨ Recommend Movies", type="primary"):
    results = recommender.recommend(
        selected_movie,
        n=n,
        genre=None if genre == "All" else genre,
        language=None if language == "All" else language,
        min_rating=min_rating,
        max_year=max_year
    )

    if results.empty:
        st.warning("No movies match the selected filters. Try relaxing the filters.")
    else:
        st.success(f"Recommendations based on **{selected_movie}**")
        for _, row in results.iterrows():
            st.markdown(
                f"### 🎬 {row['Movie_Title']}\n"
                f"**Genre:** {row['Genre']}  |  "
                f"**Rating:** ⭐ {row['Rating']:.1f}  |  "
                f"**Year:** {int(row['Release_Year'])}  |  "
                f"**Language:** {row['Language']}  |  "
                f"**Similarity:** {row['Similarity']:.2%}"
            )
            st.divider()

st.subheader("📊 Dataset Preview")
st.dataframe(recommender.df.head(20), use_container_width=True)
