#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import sklearn
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

##Side Tab:
l=["Introduction","Predict Insurance Claims"]
st.sidebar.subheader("Please select an option below")
option=st.sidebar.selectbox("Choose what you want to do:",l)

def page_1():
## Intro Tab::
    image = Image.open('insurance_claim.jpg')

## Displaying the image:
    st.image(image,use_column_width="always")

## Headers:
    st.title("Car Insurance Claim Predictive Model")
    st.header("Please input the details as requested in the next page")
    st.subheader("Let's get started...")
    
def page_2():
    data={}
## Details Tab:
    st.header("Provide details below, and predictions will be given!")

#Policy tenure, age of policyholder and car:
    tenure,age,age_car=st.columns(3)
    tenure=st.number_input("Enter your Policy Tenure (months):",step=1)
    age=st.number_input("Enter Policyholder Age (years):",step=1)
    age_car=st.number_input("Enter Age of Car (months):",step=1)
    data["Policy Tenure"]=tenure
    data["Age of Policy Holder"]=age
    data["Age of Car"]=age_car

##Area Cluster and Population Density:
    area,pop=st.columns(2)
    area=st.text_input("Enter Area Cluster: (C1-C22)")
    pop=st.text_input("Enter Population Density:")
    data["Area Cluster"]=area
    data['Population Density']=pop
                   
##Make of Car, Segment and Model of Car
    make,seg,mod=st.columns(3)
    make=st.selectbox("Select Make of Car:",[1,2,3,4,5])
    seg=st.radio('Select Car Segment',['A','B1','B2','C1','C2','Utility'])
    mod=st.selectbox("Select Car Model",["M1","M2","M3","M4","M5",'M6','M7','M8','M9','M10','M11'])
    data['Make of Car']=make
    data["Car Segment"]=seg
    data['Car Model']=mod
                
##Fuel Type
    fuel=st.radio('Select Fuel Type',['CNG','Petrol','Diesel'])
    data['Fuel Type']=fuel

##Torque
    torq_nm,torq_rpm=st.columns(2)
    torq_nm=st.slider('Select Max Torque Nm',min_value=50.0,max_value=500.0,step=1.0)
    torq_rpm=st.slider('Select Max Torque Rpm',min_value=1000.0,max_value=5000.0,step=1.0)
    data['torq_nm']=torq_nm
    data['torq_rpm']=torq_rpm
    
##Power
    power_bhp,power_rpm=st.columns(2)
    power_bhp=st.slider('Select Max Power bhp',min_value=30.0,max_value=500.0,step=1.0)
    power_rpm=st.slider('Select Max Power rpm',min_value=2000,max_value=10000,step=1000)
    data['power_bhp']=power_bhp
    data['power_rpm']=power_rpm

##Engine
    engine=st.selectbox("Select Engine Type",['F8D Petrol Engine', '1.2 L K12N Dualjet', '1.0 SCe',
       '1.5 L U2 CRDi', '1.5 Turbocharged Revotorq', 'K Series Dual jet',
       '1.2 L K Series Engine', 'K10C', 'i-DTEC',
       '1.5 Turbocharged Revotron'])
    data['engine']=engine

##Airbags and Transmission type
    transmission=st.radio('Transmission Type',['Automatic','Manual'])
    airbag=st.slider('Select Number of Airbags',1,10,1)
    data['transmission']=transmission
    data["airbags"]=airbag

##car data ESC stands for Electronic Stability Control, TPMS stands for Tyre Pressure Monitoring System,
    esc,steering,tpms,=st.columns(3)
    esc=st.radio('ESC present?',['Yes','No'])
    steering=st.radio('Adjustable steering wheel present?',['Yes','No'])
    tpms=st.radio('TPMS present?',['Yes','No'])
    data['esc']=esc
    data['steering']=steering
    data['tpms']=tpms

    sensor,camera,brakes=st.columns(3)
    sensor=st.radio('Parking Sensors present?',['Yes','No'])
    camera=st.radio('Parking Cameras present?',['Yes','No'])
    brakes=st.radio('Rear Drum Brake',['Drum','Disc'])
    data['sensor']=sensor
    data['camera']=camera
    data['brakes']=brakes

##Displacement (car cc)
    cc=st.number_input("Enter car cc here:")
    data["cc"]=cc

##Cylinders and Gear_box,steering type and turning radius
    cylinder,gear,steering,turning=st.columns(4)
    cylinder=st.number_input('Enter number of cylinders here:',step=1)
    gear=st.radio('Max Gear',['5','6'])
    steering_type=st.radio('Select Steering Type',['Power','Electric','Manual'])
    turning=st.slider('Turning Radius(m)',3.0,7.0,0.1)
    data['cylinder']=cylinder
    data['gear']=gear
    data['steering_type']=steering_type
    data['turning']=turning

##Dimensions of Car
    length,width,height,weight=st.columns(4)
    length=st.slider('Length(mm)',min_value=3000,max_value=5000,step=1)
    width=st.slider('Width(mm)',min_value=1300,max_value=2000,step=1)
    height=st.slider('Height(mm)',min_value=1300,max_value=2000,step=1)
    weight=st.slider('Weight(kg)',min_value=1000,max_value=2000,step=1)
    data['length']=length
    data['width']=width
    data['height']=height
    data['weight']=weight

##More Details of Car
    fog,wiper,washer,defogger=st.columns(4)
    fog=st.radio('Front Fog Lights present?',['Yes','No'],key=1)
    wiper=st.radio('Rear Window Wiper present?',['Yes','No'],key=2)
    washer=st.radio('Rear Window Washer present?',['Yes','No'],key=3)
    defogger=st.radio('Rear Window Wiper present?',['Yes','No'],key=4)
    data['fog']=fog
    data['wiper']=wiper
    data['washer']=washer
    data['defogger']=defogger

    assist,doorlock,central_lock,power_steer=st.columns(4)
    assist=st.radio('Brake Assist present?',['Yes','No'],key=5)
    doorlock=st.radio('Power Door Lock present?',['Yes','No'],key=6)
    central_lock=st.radio('Central Locking present?',['Yes','No'],key=7)
    power_steer=st.radio('Power Steering present?',['Yes','No'],key=8)
    data['assist']=assist
    data['doorlock']=doorlock
    data['central_lock']=central_lock
    data['power_steer']=power_steer
    
    adjust,day_night,ecw,speedalert=st.columns(4) ##ECW stands for Engine Check Warning
    adjust=st.radio('Adjustable Driver Seat present?',['Yes','No'],key=9)
    day_night=st.radio('Day Night Rear View Mirror present?',['Yes','No'],key=10)
    ecw=st.radio('ECW present?',['Yes','No'],key=11)
    speedalert=st.radio('Speed Alert present?',['Yes','No'],key=12)
    data['adjust']=adjust
    data['day_night']=day_night
    data['ecw']=ecw
    data['speedalert']=speedalert

##ncap_rating -- Safety Rating given by NCAP
    ncap=st.selectbox('Select NCAP rating',[0,1,2,3,4,5,6])
    data['ncap']=ncap
    
##Predictions Display

    cols=['policy_tenure', 'age_of_car', 'age_of_policyholder', 'area_cluster',
       'population_density', 'make', 'segment', 'model', 'fuel_type',
       'engine_type', 'airbags', 'is_esc', 'is_adjustable_steering', 'is_tpms',
       'is_parking_sensors', 'is_parking_camera', 'rear_brakes_type',
       'displacement', 'cylinder', 'transmission_type', 'gear_box',
       'steering_type', 'turning_radius', 'length', 'width', 'height',
       'gross_weight', 'is_front_fog_lights', 'is_rear_window_wiper',
       'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist',
       'is_power_door_locks', 'is_central_locking', 'is_power_steering',
       'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
       'is_ecw', 'is_speed_alert', 'ncap_rating', 'max_torque_Nm',
       'max_torque_rpm', 'max_power_bp', 'max_power_rpm'] ##list of columns in original processed dataframe
    
    input_data=[[data["Policy Tenure"],data["Age of Car"],data["Age of Policy Holder"],data["Area Cluster"],
                 data['Population Density'],data['Make of Car'],data["Car Segment"],data['Car Model'],
                 data['Fuel Type'],data['engine'],data["airbags"],data['esc'],data['steering'],data['tpms'],data['sensor'],
                 data['camera'],data['brakes'],data["cc"],data['cylinder'],data['transmission'],data['gear'],data['steering_type'],
                 data['turning'],data['length'],data['width'],data['height'],data['weight'],data['fog'],
                 data['wiper'],data['washer'],data['defogger'],data['assist'],data['doorlock'],data['central_lock'],
                 data['power_steer'],data['adjust'],data['day_night'],data['ecw'],data['speedalert'],data['ncap'],
                 data['torq_nm'],data['torq_rpm'],data['power_bhp'],data['power_rpm']]]
    
    input_data=pd.DataFrame(input_data,columns=cols)
    
    with open('insurance_claim_classifier.pkl', 'rb') as f:
        insurance_claim_classifier = pickle.load(f)

    min_threshold=0.30800962

    if st.button("Predict Insurance Claim"):
        prediction=(insurance_claim_classifier.predict_proba(input_data)[:,1]>min_threshold).astype('float')[0]
        if prediction == 0:
            st.markdown("<h2 style='text-align: center; color: green;'>This person will not claim car insurance </h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: red;'>This message will claim car insurance</h2>", unsafe_allow_html=True)

if option==l[0]:
    page_1()

if option==l[1]:
    page_2()


# In[ ]:




