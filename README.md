# AI-Based Career Recommendation System

I made this project to explore how a simple machine-learning technique can be used in a student career guidance application. A user enters skills, selects interests, and chooses a preferred type of work. The app then shows three career paths whose descriptions are most similar to that profile.

## Features

- Enter skills in your own words.
- Choose interests and a preferred work style.
- Get three suggested career paths.
- See useful skills and a short reason for each suggestion.
- Simple web interface built with Streamlit.

## How it works

The app uses **TF-IDF** to turn the student profile and career information into numerical text features. It then uses **cosine similarity** to compare the profile with each career and rank the closest matches.

The match score is a text-similarity score. It is not a probability of getting a job, and the suggestions should not be treated as a final career decision.

## Built with

- Python
- Streamlit
- Pandas
- Scikit-learn

## Run it on your computer

1. Install Python 3.10 or newer.
2. Download or clone this repository.
3. Open Terminal in the project folder.
4. (Recommended) Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, activate it with:

   ```bash
   .venv\Scripts\activate
   ```

5. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   streamlit run app.py
   ```

7. Streamlit will show a local address to open in your browser.

## Project files

```text
ai-career-recommendation-system/
├── app.py
├── careers.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## What I learned

- Building a small interactive app with Streamlit.
- Reading structured data with Pandas.
- Using TF-IDF and cosine similarity for text matching.
- Organizing a Python project for GitHub.

## Possible improvements

- Add more career paths and a larger dataset.
- Let users rate whether recommendations were useful.
- Add learning resources for each career.
- Improve recommendations with a more varied and carefully reviewed dataset.

This is an educational project, so the recommendations are basic and depend on the career descriptions included in `careers.csv`.
