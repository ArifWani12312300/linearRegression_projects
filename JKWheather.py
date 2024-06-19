#importing libraries
import streamlit as st 
import pandas as pd  
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px   
import plotly.graph_objects as go  
import numpy as np  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


#load the data
data=pd.read_csv('JK_wheatherData.csv')
data['Dates']=pd.to_datetime(data['Dates'])
data=pd.get_dummies(data,columns=['Locations'],drop_first=True)


#Define features and target variables
X=data[['Humidity','precipitation','Wind Speed','Pressure','Locations_Srinagar']]
Y=data['Temperatures']


#train the model
model=LinearRegression()
model.fit(X,Y)
y_pred=model.predict(X)
mse=mean_squared_error(Y, y_pred)

#streamlit app
st.markdown('<h1 style="color:green">Advanced Weather Prediction and Analysis in Jammu and Kashmir</h1>',unsafe_allow_html=True)
#st.title('Advanced Wheather Prediction and Analysis in Jammu and Kashmir')

#sidebar for inputer parameters
st.sidebar.markdown('<h1 style="color:green"> J And K Wheather Prediction</h1>',unsafe_allow_html=True)
st.sidebar.header('Input Parameters')
humidity=st.sidebar.slider('Humidtiy',min_value=0,max_value=100,value=50)
precipitation=st.sidebar.slider('Precipitation',min_value=0,max_value=50,value=5)
Wind_Speed=st.sidebar.slider('Wind Speed',min_value=0,max_value=50,value=10)
pressure=st.sidebar.slider('Pressure',min_value=900,max_value=1100,value=1010)
locations=st.sidebar.selectbox('locations',['Srinagar','Jammu'])

#prepare input data for prediction
input_data=pd.DataFrame({
    'Humidity':[humidity],
    'precipitation':[precipitation],
    'Wind Speed':[Wind_Speed],
    'Pressure':[pressure],
    'Locations_Srinagar':[1 if locations=='Srinagar' else 0]
})


#Display the dataset
st.markdown('<h2 style="color:lightblue;">Weather Dataset</h2>',unsafe_allow_html=True)
st.dataframe(data)


#prediction temperature
prediction=model.predict(input_data)[0]
st.markdown(f'<h2 style="color:#2E86C1;">Predicted Temperature: {prediction: .2f}</h2>',unsafe_allow_html=True)
#st.write(f'### Predicted Temperature : {prediction:.2f}')


#plotly charts
st.write('### Data Visualization')
#Line chart
fig_temp=px.line(data,x='Dates',y='Temperatures',color='Locations_Srinagar',
title='Temperature Trends in Jammu and Kashmir',
color_discrete_map={0:'blue',1:'green'})
fig_temp.update_layout(
    xaxis_title='Date',yaxis_title='temperature',
    title_font_size=24,
    template='plotly_dark'
)
st.plotly_chart(fig_temp)
#scatter plot for temperature vs humidity
fig_temp_hum=px.scatter(data,x='Humidity',y='Temperatures',color='Locations_Srinagar',
title='Temperature Vs Humidity',color_discrete_map={0:'red',1:'orange'})
fig_temp_hum.update_layout(xaxis_title='Humidity',yaxis_title='Temperature',
title_font_size=24,template='plotly_dark')
st.plotly_chart(fig_temp_hum)
#Box plot
fig_box=px.box(data,x='Locations_Srinagar',y='Temperatures',points='all',title='Temperature Distribution by  Location',
color_discrete_map={0:'purple',1:'black'})
fig_box.update_layout(xaxis_title='Location(0=Jammu,1=Srinagar)',yaxis_title='Temperature',
title_font_size=24,
template='plotly_dark')
st.plotly_chart(fig_box)

#correlation heatmap
corr_matrix=data.corr()
fig_corr=go.Figure(data=go.Heatmap(z=corr_matrix.values,
x=corr_matrix.columns,
y=corr_matrix.index,
colorscale='Viridis'))
fig_corr.update_layout(title='Correlation Heatmap',title_font_size=24,template='plotly_dark')
st.plotly_chart(fig_corr)


#time series forecasting
st.write('### Time Series Forecasting')
decomposition=seasonal_decompose(data[data['Locations_Srinagar']==1]['Temperatures'],period=1)
fig_decomp=go.Figure()
fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index,y=decomposition.trend,mode='lines',name='Trend'))
fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index,y=decomposition.seasonal,mode='lines',name='Trend'))
fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index,y=decomposition.resid,mode='lines',name='Trend'))
fig_decomp.update_layout(title='Seasonal Decomposition of Temperature in Srinagar')
st.plotly_chart(fig_decomp)

#Model Evalution
st.write('Model Evalution')
st.write(f'Mean Squared Error : {mse:.2f}')
fig_residuals=px.scatter(x=Y,y=y_pred,labels={'x':'Actual','y':'predicted'},title='Actual Vs Predicted Temperatures')
st.plotly_chart(fig_residuals)

#interactive Filtering
st.write('### Interactive Data Filtering')
start_date=st.date_input('Start Date',data['Dates'].min())
end_date=st.date_input('End Date',data['Dates'].max())
filtered_data=data[(data['Dates']>=pd.to_datetime(start_date)) & (data['Dates']<=pd.to_datetime(end_date))]
fig_filtered=px.line(filtered_data,x='Dates',y='Temperatures',color='Locations_Srinagar',title='Filtered Temperature Trends',
color_discrete_sequence=['magenta'])
st.plotly_chart(fig_filtered)


#customizable plots
st.write('### Customizable Plots')
x_axis=st.selectbox('Select X-axis variable',data.columns)
y_axis=st.selectbox('Select Y-axis variable',data.columns)
fig_custom=px.scatter(data,x=x_axis,y=y_axis,color='Locations_Srinagar',title='Customizable Scatter Plot')
st.plotly_chart(fig_custom)


#Save and Load Models
st.write('### Save and Load Models')
if st.button('Save Model'):
    import pickle
    with open('weather_model.pkl','wb') as file:
        pickle.dump(model, file)
    st.write('Model saved successfully!!!')

if st.button('Load Model'):
    with open('weather_model.pkl','rb') as file:
        loaded_model=pickle.load(file)
    loaded_prediction=loaded_model.predict(input_data)[0]
    st.write(f'### Loaded Model Prediction Temperature: {loaded_prediction: .2f}')