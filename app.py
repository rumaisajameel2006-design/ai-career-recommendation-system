import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="🎯",
    layout="centered"
)

@st.cache_data
def load_careers():
    return pd.read_csv("careers.csv")

careers = load_careers()

st.title("🎯 AI-Based Career Recommendation System")
st.write(
    "Not sure which career path suits you? Enter your skills and interests "
    "to get a few career suggestions to explore."
)
st.caption("A student project using TF-IDF and cosine similarity.")

st.subheader("Tell me about yourself")

skills = st.text_input(
    "Your skills",
    placeholder="Example: Python, communication, problem solving"
)

interests = st.multiselect(
    "What are you interested in?",
    [
        "Programming",
        "Data and numbers",
        "Artificial Intelligence",
        "Design",
        "Cybersecurity",
        "Business",
        "Helping people",
        "Research",
        "Building hardware",
        "Writing and communication"
    ]
)

work_style = st.selectbox(
    "What kind of work sounds interesting to you?",
    [
        "Choose an option",
        "Solving technical problems",
        "Finding patterns in data",
        "Creating designs and user experiences",
        "Protecting systems and information",
        "Planning and managing projects",
        "Building and testing devices",
        "Explaining ideas and helping people"
    ]
)

if st.button("Find my career matches", type="primary", use_container_width=True):
    if not skills.strip() and not interests and work_style == "Choose an option":
        st.warning("Please enter at least one skill or select an interest.")
    else:
        student_profile = " ".join(
            [skills.strip(), " ".join(interests), work_style]
        ).strip()

        career_text = (
            careers["career"] + " " +
            careers["description"] + " " +
            careers["skills"] + " " +
            careers["interests"] + " " +
            careers["work_style"]
        )

        all_text = pd.concat(
            [pd.Series([student_profile]), career_text],
            ignore_index=True
        )

        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(all_text)
        scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        results = careers.copy()
        results["match_score"] = scores
        results = results.sort_values("match_score", ascending=False).head(3)

        st.subheader("Your suggested career paths")
        st.write(
            "These suggestions are based on the words and interests you entered. "
            "Use them as a starting point, not as a final career decision."
        )

        for _, career in results.iterrows():
            percentage = round(float(career["match_score"]) * 100)
            with st.container(border=True):
                st.markdown(f"### {career['career']}")
                st.progress(min(percentage, 100))
                st.caption(f"Profile similarity: {percentage}%")
                st.write(career["description"])
                st.markdown(f"**Useful skills:** {career['skills']}")
                st.markdown(f"**Why it may fit:** {career['reason']}")

        st.info(
            "Tip: A match score shows text similarity, not your chance of getting "
            "a job or a guarantee that a career is right for you."
        )

st.divider()
st.caption("Made as a learning project by a student.")
