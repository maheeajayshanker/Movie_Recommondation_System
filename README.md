# 🎬 Movie Recommendation System

A beginner-friendly **Movie Recommendation System using Python**.  
The project follows a content-based recommendation approach using feature preprocessing, numerical conversion, feature matrices, and cosine similarity. It also provides filtering options through a Streamlit web app.

## 📁 Project Structure

```text
Movie_Recommendation_System/
│
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
└── data/
    └── movies.csv
```

## 🔧 Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Cosine Similarity

## 🧠 How It Works

1. Import libraries.
2. Load the movie dataset.
3. Understand the dataset.
4. Clean missing values.
5. Select relevant movie features.
6. Preprocess categorical and numerical features.
7. Convert categorical features into numerical form using One-Hot Encoding.
8. Create the feature matrix.
9. Calculate cosine similarity between movies.
10. Build the recommendation function.
11. Apply filters such as genre, language, rating, and release year.
12. Test recommendations.
13. Run the Streamlit application.

## ▶️ Run Locally

Open a terminal inside this project folder:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🎯 Recommendation Method

The system compares movie features including:

- Genre
- Language
- Release Year
- Rating
- Runtime

Categorical values are converted to numerical vectors, numerical values are standardized, and cosine similarity is used to find movies with similar feature profiles.

## 🔎 Filters

The Streamlit app allows users to filter recommendations by:

- Genre
- Language
- Minimum rating
- Maximum release year
- Number of recommendations

## 📊 Dataset

The included `data/movies.csv` is prepared from the supplied Movie Recommendation System dataset. It contains movie/user records with fields such as `Movie_Title`, `Genre`, `Release_Year`, `Rating`, `Number_of_Ratings`, `Runtime_Minutes`, `Language`, and `Age_Group`.

## ⚠️ Note

This is an educational content-based recommendation project. It does not use personal user-rating history to train a collaborative filtering model.
