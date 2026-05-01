
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(layout="wide")

# load css
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

df=pd.read_csv("data/data.csv")
model=pickle.load(open("models/model.pkl","rb"))

st.markdown("<h1>🌍 Urban Crime Intelligence</h1>", unsafe_allow_html=True)

# sidebar
region=st.sidebar.multiselect("Region",df.Region.unique(),default=df.Region.unique())

f=df[df.Region.isin(region)]

# KPI cards
c1,c2,c3=st.columns(3)
c1.metric("Avg Crime",round(f["Crime Index"].mean(),2))
c2.metric("Max Crime",round(f["Crime Index"].max(),2))
c3.metric("Min Crime",round(f["Crime Index"].min(),2))

# map
st.markdown("### 🌍 Global Heatmap")
fig=px.scatter_geo(f,lat="Latitude",lon="Longitude",color="Crime Index",
                   size="Crime Index",hover_name="Country")
fig.update_layout(height=700)
st.plotly_chart(fig,use_container_width=True)

# analytics
st.markdown("### 📊 Analytics")
col1,col2=st.columns(2)
col1.plotly_chart(px.line(f,x="Year",y="Crime Index",color="Country"),use_container_width=True)
col2.plotly_chart(px.bar(f,x="Country",y="Crime Index",color="Region"),use_container_width=True)

# predictor
st.markdown("### 🤖 Predictor")
year=st.slider("Year",2025,2035,2026)
lat=st.number_input("Latitude",20.0)
lon=st.number_input("Longitude",70.0)

import pandas as pd
pred=model.predict(pd.DataFrame([[year,lat,lon]],columns=["Year","Latitude","Longitude"]))[0]

st.success(f"Predicted Crime Index: {round(pred,2)}")
