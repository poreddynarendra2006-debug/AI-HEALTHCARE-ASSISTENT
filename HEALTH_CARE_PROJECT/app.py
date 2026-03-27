import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

features = ["fever","cough","headache","fatigue","vomiting","diarrhea",
            "body_pain","chills","nausea","weight_loss","high_sugar",
            "chest_pain","shortness_of_breath"]

# Page config
st.set_page_config(page_title="AI Healthcare Assistant", page_icon="🩺", layout="wide")

# Custom CSS (React-like feel)
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1c1f26;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧠 AI Healthcare")
page = st.sidebar.radio("Navigate", ["Home", "About"])

# HOME PAGE
if page == "Home":

    st.title("🩺 AI Healthcare Assistant")
    st.markdown("### Smart Disease Prediction System")

    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📝 Select Your Symptoms")

        # Multi-select instead of typing
        selected_symptoms = st.multiselect(
            "Choose symptoms",
            features
        )

        if st.button("🔍 Predict Disease"):

            if len(selected_symptoms) == 0:
                st.warning("⚠️ Please select at least one symptom")
            else:

                # Convert to vector
                input_data = [0]*len(features)
                for i, symptom in enumerate(features):
                    if symptom in selected_symptoms:
                        input_data[i] = 1

                input_df = pd.DataFrame([input_data], columns=features)

                probs = model.predict_proba(input_df)[0]
                classes = model.classes_

                top_indices = probs.argsort()[-3:][::-1]

                st.subheader("🔍 Prediction Results")

                for i in top_indices:
                    if probs[i] > 0.1:
                        st.progress(int(probs[i]*100))
                        st.write(f"**{classes[i]} → {round(probs[i]*100,2)}%**")

                best_disease = classes[top_indices[0]]

                medicines = {
                    "Flu": "Paracetamol, Rest, Fluids",
                    "Cold": "Antihistamine, Steam inhalation",
                    "Migraine": "Pain relievers, Rest in dark room",
                    "Food Poisoning": "ORS, Hydration, Light diet",
                    "Diabetes": "Metformin, Diet control, Exercise",
                    "Heart Disease": "Consult doctor immediately"
                }

                st.subheader("💊 Suggested Treatment")
                st.success(medicines.get(best_disease, "Consult doctor"))

                st.info("💡 Tip: Select more symptoms for better accuracy")
                st.warning("⚠️ Not a medical diagnosis")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("ℹ️ Instructions")
        st.write("""
        - Select symptoms from dropdown  
        - Click Predict  
        - View top diseases  
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ABOUT PAGE
elif page == "About":
    st.title("📘 About Project")

    st.markdown("""
    ### 🧠 AI Healthcare Assistant

    This project uses:
    - Machine Learning (Random Forest)
    - Symptom-based prediction
    - Streamlit UI

    ### 🎯 Features:
    - Disease prediction
    - Confidence score
    - Treatment suggestion

    ### ⚠️ Disclaimer:
    Not a replacement for doctors
    """)