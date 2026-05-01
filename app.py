
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(page_title="Urban Crime Analytics", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/crime_data.csv")

df = load_data()

st.title("🌍 Urban Crime Analytics Platform")

# Sidebar
st.sidebar.header("Filters")
countries = st.sidebar.multiselect("Country", df["Country"].unique(), default=df["Country"].unique())
year_range = st.sidebar.slider("Year Range", int(df["Year"].min()), int(df["Year"].max()), (2018,2023))

filtered = df[(df["Country"].isin(countries)) & (df["Year"].between(year_range[0], year_range[1]))]

# KPIs
col1,col2,col3 = st.columns(3)
col1.metric("Avg Crime", round(filtered["Crime Index"].mean(),2))
col2.metric("Max Crime", round(filtered["Crime Index"].max(),2))
col3.metric("Min Crime", round(filtered["Crime Index"].min(),2))

# Tabs
tab1,tab2,tab3 = st.tabs(["🌍 Map","📊 Analytics","🤖 Prediction"])

with tab1:
    fig = px.choropleth(filtered, locations="Country", locationmode="country names",
                        color="Crime Index", animation_frame="Year",
                        color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Trend")
    fig2 = px.line(filtered, x="Year", y="Crime Index", color="Country")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("City Comparison")
    fig3 = px.bar(filtered, x="City", y="Crime Index", color="Country")
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    model = pickle.load(open("models/model.pkl","rb"))
    year = st.slider("Predict Year", 2024, 2035, 2026)
    pred = model.predict([[year]])[0]
    st.success(f"Predicted Crime Index: {round(pred,2)}")
