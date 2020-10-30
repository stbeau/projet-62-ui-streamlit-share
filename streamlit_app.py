# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 23:19:56 2020

@author: sbeau
"""

from pycaret.regression import load_model, predict_model
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from joblib import load
#import os

st.beta_set_page_config(page_title="Valeur foncière", page_icon=None, layout='centered', initial_sidebar_state='auto')

#st.write(os.getcwd())

# MODEL_FILENAME = "final_xg_reg_for_streamlit_test"
# UI_DATA_FILENAME = "ui_data.pkl"
MODEL_FILENAME = "model.joblib"
UI_DATA_FILENAME = "ui_data.pkl"

#model  = load_model(MODEL_FILENAME)
model = load(MODEL_FILENAME)

with open(UI_DATA_FILENAME, "rb") as handle:
    ui_data = pickle.load(handle)


def predict(model, input_df):
    # predictions_df = predict_model(estimator=model, data=input_df)
    # predictions = predictions_df['Label'][0]
    output = model.predict(input_df)
    output = np.expm1(output[0])
    return output


# ------------------------ SIDEBAR ------------------------------

image_logo = Image.open("logo.jpg")
st.sidebar.image(image_logo, use_column_width=True) 

menu=["Prédiction des valeurs foncières","Exploration","Visuel","Le modèle"]
st.sidebar.subheader("Menu")
choix_menu = st.sidebar.selectbox("",menu)

if choix_menu == menu[0]:
    st.sidebar.subheader("Options")
    type_local=st.sidebar.selectbox("Type de propriétés",["Maison","Appartement","Dépendance","Local industriel. commercial ou assimilé"])
    #code_postal=st.sidebar.selectbox("Code postale",ui_data["code_postal"])
    #code_commune= st.sidebar.selectbox("Code commune",ui_data["code_commune"])
    #nombre_pieces_principales=st.sidebar.selectbox("Nombre de pieces principales",[0,1,2,3,4,5,6,7,8,9,10])
    nombre_pieces_principales=st.sidebar.slider("Nombre de pieces principales",min_value=1,max_value=10,value=1,step=1,format="%d")    
    surface_reelle_bati=st.sidebar.number_input("Surface réelle bati (en m^2)",value=0.0,format='%f')
    surface_terrain=st.sidebar.number_input("Surface du terrain (en m^2)",value=0.0,format='%f')
    latitude = st.sidebar.number_input("Latitude",value=0.0,format='%f')
    longitude = st.sidebar.number_input("Longitude",value=0.0,format='%f')
    
    submit = st.sidebar.button('Prédire')

    
    if submit:
        input_dic = {"latitude":latitude, "longitude":longitude,
                     "nombre_pieces_principales":nombre_pieces_principales,
                     "surface_reelle_bati":surface_reelle_bati,
                     "surface_terrain":surface_terrain, "type_local":type_local}

        input_df = pd.DataFrame([input_dic])
        output = predict(model=model,input_df=input_df)
        output = f"{output:.0f}$"

#--------------------------- PAGE 1 -----------------------------

if choix_menu == menu[0]:
    
    st.title('Prédiction des valeurs foncières en France')
    #st.subheader('Créé par : Equipe #4\n\n')
    st.subheader("")
    
    st.write("## **Résidence**")
    st.table(input_df.assign(hack="").set_index("hack"))
    st.success(f"**Valeur foncière estimée :** {output}")
    
    want_map = st.checkbox("Montrer l'emplacement")
    if want_map:
        st.map({"latitude":latitude, "longitude":longitude})
    
