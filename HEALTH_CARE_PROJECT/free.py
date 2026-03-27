import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Feature list (must match dataset)
features = ["fever","cough","headache","fatigue","vomiting","diarrhea",
            "body_pain","chills","nausea","weight_loss","high_sugar",
            "chest_pain","shortness_of_breath"]

def chatbot():
    print("\n=== 🤖 AI Healthcare Chatbot (ML) ===\n")
    print("📌 Available symptoms:")
    print(",".join(features))

    user_input = input("\nEnter symptoms (comma separated): ").lower()

    # Clean input
    user_symptoms = [s.strip() for s in user_input.split(",") if s.strip()]

    if len(user_symptoms) < 2:
        print("\n⚠️ Please enter at least 2 symptoms for better accuracy\n")

    # Convert to feature vector
    input_data = [0]*len(features)

    for i, symptom in enumerate(features):
        if symptom in user_symptoms:
            input_data[i] = 1

    # Convert to DataFrame (fix warning)
    input_df = pd.DataFrame([input_data], columns=features)

    # Get probabilities
    probs = model.predict_proba(input_df)[0]
    classes = model.classes_

    # Top 3 predictions
    top_indices = probs.argsort()[-3:][::-1]

    print("\n🤖 Bot Response:")
    print("\n🔍 Top Predictions:")
    for i in top_indices:
        print(f"{classes[i]} → {round(probs[i]*100,2)}%")

    # Best prediction
    best_index = top_indices[0]
    best_disease = classes[best_index]

    # Medicine suggestions
    medicines = {
        "Flu": "Paracetamol, Rest, Fluids",
        "Cold": "Antihistamine, Steam inhalation",
        "Migraine": "Pain relievers, Rest in dark room",
        "Food Poisoning": "ORS, Hydration, Light diet",
        "Diabetes": "Metformin, Diet control, Exercise",
        "Heart Disease": "Consult doctor immediately"
    }

    print("\n💊 Suggested Treatment:")
    print(medicines.get(best_disease, "Consult doctor"))

    print("\n💡 Tip: Enter more symptoms for better accuracy")
    print("\n⚠️ Disclaimer: This is not a medical diagnosis. Consult a doctor.\n")

# Run chatbot
chatbot()