import streamlit as st
import numpy as np
import pandas as pd
import os, joblib

# 1. Path Setup 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

# 2. Load the Champion components form previous K-means project executed 
# I'll use 'scaler_WITHOUT_outliers.pkl' because it matches the champion data explained in that project before
# Instead of joblib, use pandas:
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_WITHOUT_outliers.pkl'))
kmeans = joblib.load(os.path.join(MODEL_DIR, 'supervised_model_n_est100_rs10.pkl'))
supervised_model = joblib.load(os.path.join(MODEL_DIR, 'supervised_model_n_est100_rs10.pkl'))

st.set_page_config(page_title= 'The Housing Champion Predictor', layout= 'centered')
st.title('The California Housing: My Champion Model')
st.markdown('This app uses the **Elbow Method Champion** (Scaled data, No outliers).')

# 3. User Inputs (The 'Fuel' for your 3-feature model)
with st.container():
    st.subheader('Location & Economy')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lat = st.number_input('Latitude', value= 34.05, help= 'North-South coordinate') # This is the latitude for Los Angeles. It’s a central starting point for the dataset.
    with col2:
        lon = st.number_input('Longitude', value= -118.24, help= 'East-West coordinate') # This is the longitude for Los Angeles. It ensures the initial map marker appears in a high-density area.
    with col3:
        income = st.number_input('Median Income (MedInc)', # The dataset measures Median Income in tens of thousands. 3.5 represents an income of $35,000.
                                 min_value= 0.4999, 
                                 max_value= 8.01, 
                                 value= 3.5, 
                                 step= 0.1,
                                 help= 'Income in tens of thousands (e.g., 3.5 = $35,000). Range limited to non-outliers.') 

    # We still collect these because your SCALER needs them to process the data row
    st.divider()
    st.caption('Physical Characteristics (Required for Data Processing)')
    col4, col5 = st.columns(2)
    with col4:
        hoage = st.number_input('House Age', min_value=1.0, max_value= 52.0, value= 10.0)
    with col5:
        averoom = st.number_input('Ave. Rooms', min_value= 2.02, max_value= 8.47, value= 3.0)

# 4. I'll map the IDs to the real location shown in the scatter plot for human friendly visibility
neighborhood_profiles = {0: {'label': 'Southern California Metro (LA/San Diego)', 'desc': 'High-density urban areas in Southern California.'},
                         1: {'label': 'Northern Inland / Central Valley', 'desc': 'Agricultural and inland regions in Northern California; typically more affordable.'},
                         2: {'label': 'Premium Southern Coastal', 'desc': 'High-value coastal pockets in Orange County and San Diego.'},
                         3: {'label': 'San Francisco Bay Area', 'desc': 'The high-tech and high-cost hub of Northern California.'},
                         4: {'label': 'Greater LA Periphery', 'desc': 'Suburban transition zones surrounding the Los Angeles basin.'},
                         5: {'label': 'Central Coast & Inland', 'desc': 'Mid-state regions covering coastal towns and central communities.'}
}

if st.button("Predict Cluster"):
    # 5. The 5-feature bridge for the scaler
    input_df_5 = pd.DataFrame([[income, hoage, averoom, lat, lon]], columns=['MedInc', 'HouseAge', 'AveRooms', 'Latitude', 'Longitude'])
    
    # 6. Scale all 5 features
    scaled_data_5 = scaler.transform(input_df_5)
    
    # 7. SUBSET: Convert back to DataFrame to pick only the 3 features the MODEL wants
    scaled_df_5 = pd.DataFrame(scaled_data_5, columns=['MedInc', 'HouseAge', 'AveRooms', 'Latitude', 'Longitude'])
    model_input = scaled_df_5[['Latitude', 'Longitude', 'MedInc']]
    
    # 8. Predict
    cluster_prediction = supervised_model.predict(model_input)[0]
    
    # 9. Display the human-friendly result
    st.divider()

   # Look up the profile using the prediction
    info = neighborhood_profiles.get(cluster_prediction, {"label": "Unknown", "desc": "Data outside known clusters."})
    
    st.balloons()
    st.success(f"Analysis Successful! Targeted Cluster: {cluster_prediction}")
    st.metric("Neighborhood Profile", f"{info['label']}")
    st.write(f"**Description:** {info['desc']}")
    
    # Show exactly where they are on a map
    # Rename columns to lowercase so st.map can find them
    map_df = input_df_5[['Latitude', 'Longitude']].rename(columns={'Latitude': 'latitude','Longitude': 'longitude'})
    st.map(map_df)

with st.expander("ℹ️ About this Project"):
        st.write("This model classifies California regions into 6 clusters...")
        st.markdown("[View Source Code on GitHub](your-link-here)")