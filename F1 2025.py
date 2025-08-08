import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# CONFIGURATION
# ----------------------------
TOTAL_RACES = 23
SPRINT_FACTOR = 1 / 3
DEFAULT_COMPLETED_RACES = 14
DEFAULT_REMAINING_SPRINTS = 9

# ----------------------------
# DRIVER STANDINGS DATA
# ----------------------------
drivers_data = [
    {"name": "Oscar Piastri", "team": "McLaren", "points": 284},
    {"name": "Lando Norris", "team": "McLaren", "points": 275},
    {"name": "Max Verstappen", "team": "Red Bull Racing", "points": 187},
    {"name": "George Russell", "team": "Mercedes", "points": 172},
    {"name": "Charles Leclerc", "team": "Ferrari", "points": 151},
    {"name": "Lewis Hamilton", "team": "Ferrari", "points": 109},
    {"name": "Kimi Antonelli", "team": "Mercedes", "points": 64},
    {"name": "Alexander Albon", "team": "Williams", "points": 54},
    {"name": "Nico Hulkenberg", "team": "Kick Sauber", "points": 37},
    {"name": "Esteban Ocon", "team": "Haas", "points": 27},
    {"name": "Fernando Alonso", "team": "Aston Martin", "points": 26},
    {"name": "Lance Stroll", "team": "Aston Martin", "points": 26},
    {"name": "Isack Hadjar", "team": "Racing Bulls", "points": 22},
    {"name": "Pierre Gasly", "team": "Alpine", "points": 20},
    {"name": "Liam Lawson", "team": "Racing Bulls", "points": 20},
    {"name": "Carlos Sainz", "team": "Williams", "points": 16},
    {"name": "Gabriel Bortoleto", "team": "Kick Sauber", "points": 14},
    {"name": "Yuki Tsunoda", "team": "Red Bull Racing", "points": 10},
    {"name": "Oliver Bearman", "team": "Haas", "points": 8},
    {"name": "Franco Colapinto", "team": "Alpine", "points": 0},
    {"name": "Jack Doohan", "team": "Alpine", "points": 0},
]

# ----------------------------
# STREAMLIT APP UI
# ----------------------------
st.set_page_config(page_title="F1 2025 Predictor", layout="wide")
st.title("🏁 F1 2025 Driver Standings Predictor (AI Powered)")

st.sidebar.header("🛠️ Season Configuration")
completed_races = st.sidebar.slider("Completed Races", 1, 22, DEFAULT_COMPLETED_RACES)
remaining_races = TOTAL_RACES - completed_races
remaining_sprints = st.sidebar.slider(
    "Remaining Sprints", 0, 12, DEFAULT_REMAINING_SPRINTS
)

st.markdown(
    f"**Remaining Races:** {remaining_races} | **Remaining Sprints:** {remaining_sprints}"
)

# ----------------------------
# PREPARE TRAINING DATA
# ----------------------------
df_train = pd.DataFrame(drivers_data)
df_train["avg_pts"] = df_train["points"] / completed_races
df_train["team_encoded"] = LabelEncoder().fit_transform(df_train["team"])

X_train = df_train[["avg_pts", "team_encoded"]]
y_train = df_train["points"]

# ----------------------------
# TRAIN ML MODEL
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# MAKE PREDICTIONS
# ----------------------------
predictions = []
for driver in drivers_data:
    avg_pts = driver["points"] / completed_races if completed_races else 0
    team_enc = df_train[df_train["name"] == driver["name"]]["team_encoded"].values[0]

    predicted_race_pts = avg_pts * remaining_races
    predicted_sprint_pts = avg_pts * SPRINT_FACTOR * remaining_sprints
    final_prediction = driver["points"] + predicted_race_pts + predicted_sprint_pts

    predictions.append(
        {
            "Driver": driver["name"],
            "Team": driver["team"],
            "Current Pts": driver["points"],
            "Race Pts (AI)": round(predicted_race_pts, 1),
            "Sprint Pts (AI)": round(predicted_sprint_pts, 1),
            "Predicted Final": round(final_prediction, 1),
        }
    )

df_pred = pd.DataFrame(predictions)
df_pred.sort_values("Predicted Final", ascending=False, inplace=True)
df_pred.reset_index(drop=True, inplace=True)
df_pred.index += 1

# ----------------------------
# DISPLAY TABLE
# ----------------------------
st.subheader("🔮 Predicted Final Standings")
st.dataframe(df_pred, use_container_width=True)

# ----------------------------
# INTERACTIVE PLOT - TOP 10
# ----------------------------
st.subheader("📊 Points Prediction - Top 10 Drivers")

top_10 = df_pred.head(10).copy()
fig = px.bar(
    top_10.sort_values("Predicted Final", ascending=True),
    x="Predicted Final",
    y="Driver",
    orientation="h",
    color="Team",
    text="Predicted Final",
    hover_data=["Current Pts", "Race Pts (AI)", "Sprint Pts (AI)", "Team"],
    height=600,
)

fig.update_layout(
    xaxis_title="Predicted Final Points",
    yaxis_title="Driver",
    showlegend=True,
    plot_bgcolor="white",
    title_x=0.5,
    margin=dict(l=50, r=30, t=50, b=30),
)

fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)
