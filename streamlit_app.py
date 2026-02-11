import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG & DATA ---
st.set_page_config(page_title="Iris Smart Predictor", page_icon="🌸")

SPECIES_DATA = {
    "setosa": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/320px-Kosaciec_szczecinkowaty_Iris_setosa.jpg",
    "versicolor": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg",
    "virginica": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/320px-Iris_virginica.jpg"
}
df = px.data.iris()

df_means = df.groupby('species').mean(numeric_only=True)

# --- UI STYLING ---
st.markdown("""
    <style>
    .img-container img { height: 200px; width: 100%; object-fit: cover; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🌸 Iris Species Predictor")
cols = st.columns(3)
for i, (name, url) in enumerate(SPECIES_DATA.items()):
    cols[i].markdown(f'<div class="img-container"><img src="{url}"></div>', unsafe_allow_html=True)
    cols[i].caption(f"Iris {name.title()}")

st.divider()

# --- APP LOGIC ---
tab1, tab2 = st.tabs(["Prediction", "Data Analysis"])

with tab1:
    st.header("Predict the Species")
    
    # Grouping inputs into columns efficiently
    c1, c2 = st.columns(2)
    sl = c1.slider("Sepal Length (cm)", 4.0, 12.0, 5.1)
    sw = c1.slider("Sepal Width (cm)", 2.0, 7.0, 3.5)
    pl = c2.slider("Petal Length (cm)", 1.0, 11.0, 1.4)
    pw = c2.slider("Petal Width (cm)", 0.1, 5.0, 0.2)

    if st.button("🔍 Predict Species", type="primary"):
        try:
            with st.spinner('Analyzing...'):
                payload = {"sepal_length": sl, "sepal_width": sw, "petal_length": pl, "petal_width": pw}
                resp = requests.post("https://iris-inference-service.onrender.com/predict", json=payload)
                resp.raise_for_status()
                st.session_state.last_prediction = resp.json()
        except Exception as e:
            st.error(f"Error: {e}")

    # Display Results
    if "last_prediction" in st.session_state:
        res = st.session_state.last_prediction
        name = res['species'].split('-')[-1].lower() # Extracting 'setosa' from 'Iris-setosa'
        
        st.divider()
        col_img, col_txt = st.columns([1, 2])
        col_img.image(SPECIES_DATA.get(name), width='stretch')
        
        with col_txt:
            st.success(f"### Result: {name.title()}")
            st.metric("Confidence", f"{res['probability']:.2%}")
            st.progress(res['probability'])

        # Radar Chart
        categories = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[sl, sw, pl, pw], theta=categories, fill='toself', name='Input', line_color='red'))
        
        if name in df_means.index:
            fig.add_trace(go.Scatterpolar(r=df_means.loc[name, categories], theta=categories, fill='toself', name=f'Avg {name}'))

        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 12])), height=350)
        st.plotly_chart(fig, width='stretch')

# TAB 2: VISUALIZATION

with tab2:
    
    st.header("Understand the Data")

    st.subheader("Explore the Dataset")
    st.write("This is the famous Iris dataset containing 150 samples.")
    
    if st.toggle("Show Raw Data"):
        st.dataframe(df)

        st.caption(f"Total rows: {len(df)}")
        st.write(df.describe()) 

    st.divider()
    
   #INTERACTIVE 3D PLOT 
    st.subheader("1. 3D Cluster View")
    st.write("Select which 3 features to map to the X, Y, and Z axes.")

    # 1. Get the list of numerical columns (exclude 'species' and 'species_id')
    # This automatically finds: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    feature_cols = df.columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ['species', 'species_id']]

    # 2. create 3 Columns for the Selectboxes so they look nice side-by-side
    x_col, y_col, z_col = st.columns(3)

    with x_col:
        x_axis = st.selectbox("X Axis", feature_cols, index=0) 
    with y_col:
        y_axis = st.selectbox("Y Axis", feature_cols, index=1) 
    with z_col:
        z_axis = st.selectbox("Z Axis", feature_cols, index=3) 

    fig_3d = px.scatter_3d(
        df, 
        x=x_axis, 
        y=y_axis, 
        z=z_axis,
        color='species',
        symbol='species',
        opacity=0.7,
        size_max=10
    )
    
    # Update layout to be cleaner
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    
    st.plotly_chart(fig_3d, width='stretch')
    
    # Graph 2: Comparison Scatter
    st.subheader("2. Petal Dimensions Comparison")
    st.write("Notice how Setosa (Blue) is much smaller than the others.")
    fig_scatter = px.scatter(
        df, 
        x="petal_length", 
        y="petal_width", 
        color="species",
        size="sepal_length",
        hover_data=['sepal_width']
    )
    st.plotly_chart(fig_scatter, width='stretch')
    
    # Graph 2: Comparison Scatter
    st.subheader("2. Sepal Dimensions Comparison")
    fig_scatter = px.scatter(
        df, 
        x="sepal_length", 
        y="sepal_width", 
        color="species",
        size="petal_length",
        hover_data=['petal_width']
    )
    st.plotly_chart(fig_scatter, width='stretch')
    
    # Graph 3: Distribution
    st.subheader("3. Distribution of Features")
    feature_to_plot = st.selectbox("Select a feature to compare:", 
                                  ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    
    fig_hist = px.histogram(
        df, 
        x=feature_to_plot, 
        color="species", 
        barmode="overlay",
        marginal="box" 
    )
    st.plotly_chart(fig_hist, width='stretch')