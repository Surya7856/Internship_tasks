import streamlit as st
import pandas as pd
import numpy as np
import pickle

file1=open('pipe.pkl','rb')
rf=pickle.load(file1)
file1.close()

st.set_page_config(page_title="Laptop price prediction")
data=pd.read_csv('laptop_data_cleaned.csv')


st.title('Laptop Price Predictor')

company=st.selectbox('Brand',data['Company'].unique())

type=st.selectbox('Type',data['TypeName'].unique())

ram=st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

weight=st.number_input('Weight of the laptop(in Kg)')

os = st.selectbox('OS', data['Os'].unique())

touchscreen=st.selectbox('Touchscreen',['No','Yes'])

ips=st.selectbox('IPS',['No','Yes'])

screen_size=st.number_input('Screen Size(in Inches)')

resolution=st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

cpu=st.selectbox('CPU',data['Cpu_brand'].unique())

hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', data['Gpu_brand'].unique())

if st.button('Predict Price'):
    ppi=None
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
    if ips=='Yes':
        ips=1
    else:
        ips=0
    X_resol=int(resolution.split('x')[0])
    Y_resol=int(resolution.split('x')[1])
    ppi = ((X_resol**2)+(Y_resol**2))**0.5/(screen_size)
    query = np.array([company, type, ram,weight,
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu,os])
    query=query.reshape(1,12)
    query=query.astype('object')
    prediction=int(np.exp(rf.predict(query)[0]))
    st.title("Predicted price for this laptop could be between "+str(prediction-1000)+" to "+str(prediction+1000))

st.sidebar.title("About")
st.sidebar.write("This app is designed to help you predict the price of a laptop based on various specifications.\nSimply enter the details of your desired laptop, and our machine learning model will generate an estimated price.\nWhether you're a student,professional, or just someone looking for a new laptop, this app can help you save money and find the perfect laptop for your needs.")



                        





