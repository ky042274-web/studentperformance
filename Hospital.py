import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Hospital Disease Prediction",
    page_icon="🏥",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fff3e0, #fdebd0);
        color: black;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: black !important;
    }

    .stButton > button {
        background-color: #ffb74d;
        color: black;
        border-radius: 12px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #ffa726;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------ HEADER PROMPT ------------------
st.markdown(
    """
    <h1>🏥 Hospital Disease Prediction System</h1>
    <p style="text-align:center; font-size:18px; color:#333;">
    Enter patient vitals and let AI assist in predicting possible diseases.
    </p>
    """,
    unsafe_allow_html=True
)

# ------------------ LOAD DATA ------------------
df = pd.read_csv("hospital_data_100.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df)

# ------------------ GROUPBY ------------------
st.subheader("📈 Average Values by Disease")
st.write(df.groupby("Disease").mean())

# ------------------ MODEL ------------------
X = df[["Fever", "BP", "Sugar"]]
Y = df["Disease"]

le = LabelEncoder()
y_enc = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# ------------------ ACCURACY ------------------
pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, pred)

st.subheader("✅ Model Accuracy")
st.success(f"Accuracy: {accuracy * 100:.2f}%")

# ------------------ PREDICTION SECTION ------------------
st.subheader("🧪 Predict Disease")

fever = st.number_input("🌡 Fever Level", min_value=0)
bp = st.number_input("🩸 Blood Pressure", min_value=0)
sugar = st.number_input("🍬 Sugar Level", min_value=0)

if st.button("🔍 Predict Disease"):
    new_data = [[fever, bp, sugar]]
    prediction = model.predict(new_data)
    disease = le.inverse_transform(prediction)

    st.info(f"🩺 **Predicted Disease:** {disease[0]}")



