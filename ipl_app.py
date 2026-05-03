import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Data load
df = pd.read_csv("matches.csv")
df = df.dropna(subset=['winner'])

# Venue fix
venue_mapping = {
    'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',
    'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
    'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
    'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
    'Eden Gardens, Kolkata': 'Eden Gardens',
    'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
    'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association IS Bindra Stadium',
    'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
    'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium',
    'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium',
    'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
    'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
    'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
    'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium',
    'Barsapara Cricket Stadium, Guwahati': 'Barsapara Cricket Stadium',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 'Ekana Cricket Stadium',
    'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
    'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur': 'Maharaja Yadavindra Singh International Cricket Stadium',
}
df['venue'] = df['venue'].replace(venue_mapping)

# Features
df['toss_match_winner'] = (df['toss_winner'] == df['winner']).astype(int)

home_teams_dict = {
    'Chennai Super Kings': 'Chennai',
    'Mumbai Indians': 'Mumbai',
    'Royal Challengers Bangalore': 'Bengaluru',
    'Royal Challengers Bengaluru': 'Bengaluru',
    'Kolkata Knight Riders': 'Kolkata',
    'Rajasthan Royals': 'Jaipur',
    'Sunrisers Hyderabad': 'Hyderabad',
    'Delhi Daredevils': 'Delhi',
    'Delhi Capitals': 'Delhi',
    'Kings XI Punjab': 'Mohali',
    'Punjab Kings': 'Mohali',
    'Gujarat Titans': 'Ahmedabad',
    'Lucknow Super Giants': 'Lucknow'
}

df['team1_home'] = (df['team1'].map(home_teams_dict) == df['city']).astype(int)
df['team2_home'] = (df['team2'].map(home_teams_dict) == df['city']).astype(int)
df['team1_won'] = (df['winner'] == df['team1']).astype(int)

# Encoders
le_team = LabelEncoder()
le_venue = LabelEncoder()
le_toss = LabelEncoder()

ml_df = df[['team1', 'team2', 'venue', 'toss_winner',
            'toss_decision', 'team1_home', 'team2_home',
            'toss_match_winner', 'team1_won']].copy()

all_teams = pd.concat([ml_df['team1'], ml_df['team2'],
                       ml_df['toss_winner']]).unique()
le_team.fit(all_teams)

ml_df['team1'] = le_team.transform(ml_df['team1'])
ml_df['team2'] = le_team.transform(ml_df['team2'])
ml_df['toss_winner'] = le_team.transform(ml_df['toss_winner'])
ml_df['venue'] = le_venue.fit_transform(ml_df['venue'])
ml_df['toss_decision'] = le_toss.fit_transform(ml_df['toss_decision'])

X = ml_df.drop('team1_won', axis=1)
y = ml_df['team1_won']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# UI
teams = sorted(df['team1'].unique())
venues = sorted(df['venue'].unique())

st.set_page_config(page_title="IPL Winner Predictor", page_icon="🏏")
st.title("🏏 IPL Match Winner Predictor")
st.subheader("AI se poocho — kaun jeetega!")

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1 select karo", teams)
with col2:
    team2 = st.selectbox("Team 2 select karo", teams)

venue = st.selectbox("Venue select karo", venues)

col3, col4 = st.columns(2)
with col3:
    toss_winner = st.selectbox("Toss kisne jeeta?", [team1, team2])
with col4:
    toss_decision = st.selectbox("Toss decision", ["bat", "field"])

if st.button("🚀 Predict Winner!"):
    if team1 == team2:
        st.error("Bhai ek hi team ke saath match nahi hota! 😄")
    else:
        try:
            t1 = le_team.transform([team1])[0]
            t2 = le_team.transform([team2])[0]
            tw = le_team.transform([toss_winner])[0]
            v  = le_venue.transform([venue])[0]
            td = le_toss.transform([toss_decision])[0]

            city = home_teams_dict.get(team1, 'Neutral')
            t1_home = 1 if home_teams_dict.get(team1) == city else 0
            t2_home = 1 if home_teams_dict.get(team2) == city else 0
            toss_match = 1 if toss_winner == team1 else 0

            features = pd.DataFrame(
                [[t1, t2, v, tw, td, t1_home, t2_home, toss_match]],
                columns=X.columns)

            prob = model.predict_proba(features)[0]
            winner = team1 if prob[1] > 0.5 else team2
            confidence = max(prob) * 100

            st.success(f"🏆 Predicted Winner: **{winner}**")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.progress(int(prob[1] * 100))

            col5, col6 = st.columns(2)
            with col5:
                st.metric(f"{team1} Win Chance", f"{prob[1]*100:.1f}%")
            with col6:
                st.metric(f"{team2} Win Chance", f"{prob[0]*100:.1f}%")

        except Exception as e:
            st.error(f"Error: {e}")

print("App file ban gayi!")
