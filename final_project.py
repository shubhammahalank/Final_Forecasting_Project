# Importing Libraries
import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Importing Dataframes
calendar = pd.read_csv('calendar.csv')
sales = pd.read_csv('sales_train_validation.csv')
sell_prices = pd.read_csv('sell_prices.csv')
submission = pd.read_csv('sample_submission.csv')

# Streamlit Credentials
# from PIL import Image
# img = Image.open("walmart_logo.png")
# st.image(img,width=700,use_column_width=True)
st.title("M5 Forecasting - Accuracy") # Title
st.header("Forecasting of sales of products at Walmart")
#st.subheader("")
#st.text("")
#st.markdown("")
#st.success("")
#st.info("")
#st.warning("")
#st.error("")
#st.exception("")

#Checkbox
# if st.checkbox("Show/Hide"):
# 	st.text("This is hidden text")

#Radio Button
# state = st.radio("Select State?",("California","Texas","Wisconsin"))

#Select Box
# state = st.selectbox("Select State?",["California","Texas","Wisconsin"])

#Multiselect
state = st.multiselect("Select State:",("California","Texas","Wisconsin"))
# st.write("You selected", len(state), "state/s")
store_id = st.multiselect("Select Store_id:",("CA_1","CA_2","CA_3","CA_4","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"))

#Slider
# level = st.slider("What is your level?",1,5)

#Buttons
# st.button("Simple Button")
# if st.button("Simple Button"):
# 	st.text("Some text")

#Text Input
# firstname = st.text_input("Enter Your Firstname","Type here...")
# if st.button("Submit"):
# 	result = firstname.title()
# 	st.success(result)

# message = st.text_area("Enter Your message","Type here...")
# if st.button("Submit"):
# 	result = message.title()
# 	st.success(result)

#Date Input
# today = st.date_input("Today is",datetime.datetime.now())
#Time
# the_time = st.time_input("The time is",datetime.time())

#Display raw code
# with st.echo():
# 	# This will show as a comment
# 	import pandas as pd
# 	df = pd.DataFrame()

# Progress Bar
# import time
# my_bar = st.progress(0)
# for p in range(10):
# 	my_bar.progress(p+1)

# # Spinner
# with st.spinner("Loading.."):
# 	time.sleep(5)
# st.success("Finished!")

# Balloons
# st.balloons()

# Sidebar
# st.sidebar.header("About")
# st.sidebar.text("This is sidebar tut")

# Functions
# @st.cache
# def run_func():
# 	return range(100)
# st.write(run_func())

# Plot
# st.pyplot()

# DataFrames
# st.dataframe(calendar)

# Tables
# st.table(df)






