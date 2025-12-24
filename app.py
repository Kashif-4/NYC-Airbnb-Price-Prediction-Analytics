import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(page_title="NYC Airbnb Analytics", layout="wide")

st.title("🗽 NYC Airbnb Analytics & Price Predictor")
st.markdown("Explore Airbnb listings in New York City and predict rental prices.")

# --- Data Loading ---
@st.cache_data
def load_data():
    if not os.path.exists("AB_NYC_2019.csv"):
        st.error("Dataset not found. Please run the analysis notebook first to download it.")
        return None
    df = pd.read_csv("AB_NYC_2019.csv")
    # Basic cleaning matching the notebook
    df = df[df['price'] < 1000]
    df = df[df['price'] > 0]
    return df

@st.cache_resource
def load_model():
    if not os.path.exists("nyc_airbnb_model.pkl"):
        st.warning("Model not found. Please run the ML notebook to generate 'nyc_airbnb_model.pkl'.")
        return None
    return joblib.load("nyc_airbnb_model.pkl")

df = load_data()
artifacts = load_model()

if df is not None:
    # --- Sidebar ---
    st.sidebar.header("Filter Listings")
    
    boroughs = df['neighbourhood_group'].unique()
    selected_boroughs = st.sidebar.multiselect("Select Boroughs", boroughs, default=boroughs)
    
    price_range = st.sidebar.slider("Price Range ($)", 0, 1000, (0, 500))
    
    room_types = df['room_type'].unique()
    selected_rooms = st.sidebar.multiselect("Room Type", room_types, default=room_types)
    
    # Filter Logic
    filtered_df = df[
        (df['neighbourhood_group'].isin(selected_boroughs)) & 
        (df['price'].between(price_range[0], price_range[1])) &
        (df['room_type'].isin(selected_rooms))
    ]
    
    st.sidebar.write(f"Showing {len(filtered_df)} listings")

    # --- Main Dashboard ---
    
    # Row 1: Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Price", f"${filtered_df['price'].mean():.2f}")
    col2.metric("Total Listings", len(filtered_df))
    if not filtered_df.empty:
        col3.metric("Most Expensive", f"${filtered_df['price'].max()}")
    
    # Row 2: Map
    st.subheader("📍 Listings Map")
    if not filtered_df.empty:
        st.map(filtered_df[['latitude', 'longitude']])
    else:
        st.info("No listings match your filters.")
        
    # Row 3: Price Prediction
    st.divider()
    st.subheader("💰 Price Predictor")
    st.markdown("Enter details to predict the nightly rate.")
    
    if artifacts:
        model = artifacts['model']
        le_group = artifacts['le_group']
        le_room = artifacts['le_room']
        le_neigh = artifacts['le_neigh']
        
        c1, c2, c3 = st.columns(3)
        input_borough = c1.selectbox("Borough", le_group.classes_)
        input_room = c2.selectbox("Room Type", le_room.classes_)
        input_nights = c3.number_input("Minimum Nights", min_value=1, value=1)
        
        # Neighbourhood needs to be filtered by borough ideally, but for MVP we list all or top
        # To handle 'unknown' regression labels if user picks something rarely seen, we handle exceptions or just list all
        input_neigh_name = st.selectbox("Neighbourhood", le_neigh.classes_)
        
        input_availability = st.slider("Availability (Days/Year)", 0, 365, 100)
        input_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
        
        if st.button("Predict Price"):
            try:
                # Encode inputs
                grp_enc = le_group.transform([input_borough])[0]
                room_enc = le_room.transform([input_room])[0]
                neigh_enc = le_neigh.transform([input_neigh_name])[0]
                
                # Create Feature Vector (order must match training)
                # Training cols: [neighbourhood_group, neighbourhood, latitude, longitude, room_type, minimum_nights, 
                #                 number_of_reviews, reviews_per_month, calculated_host_listings_count, availability_365]
                
                # Note: We are missing Lat/Lon for the specific neighborhood chosen by user.
                # Project simplification: Use mean lat/lon of that neighborhood from dataset
                neigh_data = df[df['neighbourhood'] == input_neigh_name]
                if not neigh_data.empty:
                    mean_lat = neigh_data['latitude'].mean()
                    mean_lon = neigh_data['longitude'].mean()
                else:
                    mean_lat = 40.7128
                    mean_lon = -74.0060
                    
                # Other missing feats: reviews_per_month, host_listings_count (use median/mode)
                rev_per_month = df['reviews_per_month'].mean()
                host_count = 1
                
                features = np.array([[
                    grp_enc, neigh_enc, mean_lat, mean_lon, room_enc, 
                    input_nights, input_reviews, rev_per_month, host_count, input_availability
                ]])
                
                prediction = model.predict(features)[0]
                st.success(f"Estimated Price: **${prediction:.2f}** / night")
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.info("Model artifacts not available. Run the notebook first.")

else:
    st.info("Loading data...")
