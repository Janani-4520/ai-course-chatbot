import streamlit as st
import json
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

patterns = []
tags = []
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    tag = tags[index]
    return random.choice(responses[tag])

st.title("🎓 AI Course Guidance Chatbot")

user_input = st.text_input("Ask your question:")

if user_input:
    response = chatbot_response(user_input)
    st.write("🤖 Bot:", response)
