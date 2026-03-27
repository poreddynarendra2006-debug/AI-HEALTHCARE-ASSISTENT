import streamlit as st
import pandas as pd
import pickle
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = pickle.load(open(model_path, "rb"))

features = ["fever","cough","headache","fatigue","vomiting","diarrhea",
            "body_pain","chills","nausea","weight_loss","high_sugar",
            "chest_pain","shortness_of_breath"]

st.set_page_config(page_title="AI Healthcare Chatbot", page_icon="🧠", layout="wide")

# 🎨 Vibrant CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    color: white;
}

.user-msg {
    background: linear-gradient(135deg, #43cea2, #185a9d);
    padding: 12px;
    border-radius: 15px;
    color: white;
    text-align: right;
    margin: 10px 0;
}

.bot-msg {
    background: #2c3e50;
    padding: 12px;
    border-radius: 15px;
    color: white;
    margin: 10px 0;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #ff7e5f, #feb47b);
    border-radius: 10px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧠 AI Healthcare")
page = st.sidebar.radio("Navigate", ["Home", "About"])

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🏠 HOME
if page == "Home":

    st.title("🧠 AI Healthcare Chatbot")
    st.write("👋 Hello! How can I assist you today?")

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("🩺 Symptom Checker")
        st.button("📍 Find a Doctor")
    with col2:
        st.button("💊 Medication Info")
        st.button("❤️ Health Tips")

    st.markdown("---")

    # Show chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

    # Input
    user_input = st.text_input("Type your symptoms (comma separated)")

    if st.button("Send"):

        if user_input.strip() != "":
            st.session_state.messages.append({"role": "user", "content": user_input})

            user_symptoms = [s.strip().lower() for s in user_input.split(",")]

            # Convert to vector
            input_data = [0]*len(features)
            for i, symptom in enumerate(features):
                if symptom in user_symptoms:
                    input_data[i] = 1

            input_df = pd.DataFrame([input_data], columns=features)

            probs = model.predict_proba(input_df)[0]
            classes = model.classes_

            disease = classes[probs.argmax()]

            # 💊 Medicines
            medicines = {
                "Flu": "Paracetamol, Rest, Fluids",
                "Cold": "Antihistamine, Steam inhalation",
                "Migraine": "Pain relievers, Rest in dark room",
                "Food Poisoning": "ORS, Hydration",
                "Diabetes": "Metformin, Diet control",
                "Heart Disease": "Consult doctor immediately"
            }

            # 💡 Health Tips
            tips = {
                "Flu": "Drink warm fluids, take rest, and avoid cold exposure.",
                "Cold": "Stay hydrated, take steam inhalation, and rest well.",
                "Migraine": "Avoid bright lights, rest in a quiet room.",
                "Food Poisoning": "Drink ORS, avoid oily food, maintain hygiene.",
                "Diabetes": "Maintain healthy diet, exercise regularly.",
                "Heart Disease": "Avoid stress, follow healthy diet, regular checkups."
            }

            response = f"""
🩺 **Disease:** {disease}  
💊 **Treatment:** {medicines.get(disease)}  
💡 **Health Tip:** {tips.get(disease)}
"""

            st.session_state.messages.append({"role": "bot", "content": response})

            st.rerun()

# 📘 ABOUT
elif page == "About":
    st.title("📘 About")
    st.write("AI Healthcare Chatbot with vibrant UI and ML-based predictions.")
