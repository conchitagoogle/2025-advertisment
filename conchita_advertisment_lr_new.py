import streamlit as st
from datetime import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs  as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

import numpy as np



st.title("Advertisement and Sales Data")

st.markdown("""
	The data set contains information about money spent on advertisement and their generated sales. Money
	was spent on TV, radio and newspaper ads.
	## Problem Statement
	Sales (in thousands of units) for a particular product as a function of advertising budgets (in thousands of
	dollars) for TV, radio, and newspaper media. Suppose that in our role as statistical consultants we are
	asked to suggest.
	Here are a few important questions that you might seek to address:
	- Is there a relationship between advertising budget and sales?
	- How strong is the relationship between the advertising budget and sales?
	- Which media contribute to sales?
	- How accurately can we estimate the effect of each medium on sales?
	- How accurately can we predict future sales?
	- Is the relationship linear?
	We want to find a function that given input budgets for TV, radio and newspaper predicts the output sales
	and visualize the relationship between the features and the response using scatter plots.
	The objective is to use linear regression to understand how advertisement spending impacts sales.
	
	### Data Description
	TV
	Radio
	Newspaper
	Sales
""")
st.sidebar.title("Operations on the Dataset")


w1 = st.sidebar.checkbox("show table", False)
plothist= st.sidebar.checkbox("show hist plots", False)
trainmodel= st.sidebar.checkbox("Train model", False)
linechart=st.sidebar.checkbox("Linechart",False)
#st.write(w1)


@st.cache_data
def read_data():
    return pd.read_csv("Advertising.csv")[["TV","radio","newspaper","sales"]]

df=read_data()


if w1:
    st.dataframe(df,width=2000,height=500)
if linechart:
	st.subheader("Line chart")
	st.line_chart(df)
if plothist:
    st.subheader("Distributions of each columns")
    options = ("TV","radio","newspaper","sales")
    sel_cols = st.selectbox("select columns", options,1)
    st.write(sel_cols)
    fig = go.Histogram(x=df[sel_cols],nbinsx=50)
    st.plotly_chart([fig])
    



if trainmodel:
	st.header("Modeling")
	y=df.sales
	X=df[["TV","radio","newspaper"]].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	lrgr = LinearRegression()
	lrgr.fit(X_train,y_train)
	pred = lrgr.predict(X_test)

	mse = mean_squared_error(y_test,pred)
	rmse = sqrt(mse)

	st.markdown(f"""
	Linear Regression model trained :
		- MSE:{mse}
		- RMSE:{rmse}
	""")
	st.success('Model trained successfully')


