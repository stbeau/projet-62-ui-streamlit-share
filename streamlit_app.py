# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 23:19:56 2020

@author: sbeau
"""

from pycaret.regression import load_model, predict_model
#from collections import OrderedDict
#from IPython.display import HTML
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
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


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
    surface_reelle_bati=st.sidebar.number_input("Surface réelle bati (en m^2)",value=0.0,format='%f',step=1.0)
    surface_terrain=st.sidebar.number_input("Surface du terrain (en m^2)",value=0.0,format='%f')
    latitude = st.sidebar.number_input("Latitude",value=0.0,format='%f')
    longitude = st.sidebar.number_input("Longitude",value=0.0,format='%f')
    
    submit = st.sidebar.button('Prédire')

    input_dic = {"latitude":"", "longitude":"",
                 "nombre_pieces_principales":"",
                 "surface_reelle_bati":"",
                 "surface_terrain":"", "type_local":""}
    input_df = pd.DataFrame([input_dic])
    
    output = 0.0
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
    
    st.checkbox("Montrer des comparables")
    
  colnames = ["latitude":latitude, "longitude":longitude, "nombre_pieces_principales","surface_reelle_bati","surface_terrain","type_local"]


# GarageArea=st.sidebar.number_input("Enter area of Garage (in Sqft)",value=0.0,format='%f',step=1.0)
# GarageCars=st.sidebar.number_input("Number of Cars to be accomodated in garage",min_value=1.0,max_value=10.0,step=1.0,format='%f')
# TotRmsAbvGrd=st.sidebar.number_input("Enter number of Rooms",min_value=1,max_value=10,format='%d')
# years=tuple([i for i in range(1872,2011)])
# YearBuilt=st.sidebar.selectbox("Select the overall quality(10 being 'Very Excellent' and 1 being 'very poor')",years)
# remyears=tuple([i for i in range(1950,2011)])
# YearRemodAdd=st.sidebar.selectbox("Select Remodel date (same as construction date if no remodeling or additions)",remyears)
# garyears=tuple([i for i in range(1872,2011)])
# garyears=tuple(map(float,garyears))
# GarageYrBlt=st.sidebar.selectbox("Select year in which Garage was built)",garyears)
# MasVnrArea=st.sidebar.number_input("Masonry veneer area (in Sqft)",value=0.0,format='%f',step=1.0)
# Fireplaces=st.sidebar.number_input("Select number of FirePlaces",min_value=1,max_value=10,format='%d')
# BsmtFinSF1=st.sidebar.number_input("Enter Basement Finished Area(in Sqft)",value=0,format='%d')
# submit = st.sidebar.button('Predict')