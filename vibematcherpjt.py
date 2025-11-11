import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import time

# -----------------------------
# 🎨 Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Vibe Matcher 2025", page_icon="✨", layout="wide")

# -----------------------------
# 🌈 Stylish CSS
# -----------------------------
st.markdown("""
<style>
/* Background gradient suitable for black and white text */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f0f0, #a0c4ff, #e0e0e0);
    background-size: cover;
}

/* Input field style */
.stTextInput>div>div>input {
    color: white !important;                  /* typed text */
    background-color: #000000 !important;     /* black input background */
    border: 1px solid #ffffff !important;     /* white border */
    border-radius: 12px;
    padding: 0.6rem;
    font-size: 16px;
}

/* Placeholder text */
.stTextInput>div>div>input::placeholder {
    color: #cccccc !important;                /* placeholder color */
}

/* Button style */
.stButton>button {
    border-radius: 12px;
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #1d4ed8;
    transform: scale(1.03);
}

/* Match card */
.match-card {
    background-color: rgba(0,0,0,0.7);
    padding: 1.2rem;
    border-radius: 16px;
    color: white;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
}
.match-card h3 {
    margin: 0;
    color: #93c5fd;
}
.match-card p {
    margin: 0.2rem 0;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Page Title
# -----------------------------
st.markdown('<h1 style="text-align:center;color:#1e293b;">💫 Vibe Matcher 2025</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#1e293b;">Find your perfect fashion match based on your vibe ✨</p>', unsafe_allow_html=True)

# -----------------------------
# Sample Data
# -----------------------------
data = [
    {"name": "Boho Dress", "desc": "Flowy, earthy tones for festival vibes", "tags": ["boho", "relaxed"]},
    {"name": "Urban Jacket", "desc": "Sleek streetwear for energetic city life", "tags": ["urban", "chic"]},
    {"name": "Cozy Sweater", "desc": "Soft and warm, perfect for relaxed days", "tags": ["cozy", "casual"]},
    {"name": "Elegant Blazer", "desc": "Chic formal wear for office and parties", "tags": ["formal", "chic"]},
    {"name": "Sporty Sneakers", "desc": "Comfortable shoes for energetic outdoor activities", "tags": ["sporty", "energetic"]},
    {"name": "Beach Shorts", "desc": "Light and airy for sunny beach days", "tags": ["casual", "summer"]},
]
df = pd.DataFrame(data)

# -----------------------------
# OpenAI Embeddings Setup
# -----------------------------
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))

@st.cache_resource
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

# Precompute embeddings for product descriptions
if 'embedding' not in df.columns:
    df['embedding'] = df['desc'].apply(get_embedding)

# -----------------------------
# Find Top Match Function
# -----------------------------
def find_top_match(query, df, threshold=0.8):
    query_emb = get_embedding(query)
    sims = cosine_similarity([query_emb], list(df['embedding']))[0]
    df['similarity'] = sims

    top_match = df.sort_values(by='similarity', ascending=False).head(1)
    if top_match['similarity'].values[0] < threshold:
        return None
    return top_match.iloc[0]

# -----------------------------
# User Input
# -----------------------------
st.markdown('<p style="color:#1e293b; font-size:16px;">✨ Enter your vibe query:</p>', unsafe_allow_html=True)
query = st.text_input("", placeholder="Type your vibe here...")

if st.button("Find My Match"):
    if query.strip() == "":
        st.error("❌ Please enter a vibe query!")
    else:
        start_time = time.time()
        top_match = find_top_match(query, df)
        end_time = time.time()
        latency = round(end_time - start_time, 2)

        if top_match is None:
            st.warning("⚠️ No match found! Try rephrasing your vibe.")
        else:
            st.markdown(f"""
            <div class="match-card">
                <h3>{top_match['name']} 🪄</h3>
                <p>{top_match['desc']}</p>
                <p style='font-size:13px;'>Tags: {', '.join(top_match['tags'])}</p>
                <p style='font-size:12px;'>Similarity: {top_match['similarity']:.3f}</p>
                <p style='font-size:12px;'>⏱️ Response Time: {latency}s</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown('<p style="text-align:center;color:gray;font-size:13px;">© 2025 VibeMatcher AI | Built with ❤️ using Streamlit & OpenAI</p>', unsafe_allow_html=True)
