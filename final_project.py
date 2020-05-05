# Importing Libraries
import numpy as np
import pandas as pd
import datetime
import wget

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

import tqdm as tqdm
import statsmodels.api as sm
# from fbprophet import Prophet

# Importing Dataframes
@st.cache
def  loadfile():
	sales_train_validation = wget.download("https://s3.us-east-2.amazonaws.com/data.forecasting/sample_submission.csv")
	# calendar = pd.read_csv('calendar.csv')
	sales = pd.read_csv(sales_train_validation)
	# sell_prices = pd.read_csv('sell_prices.csv')
	# submission = pd.read_csv('sample_submission.csv')
	return sales

sales = loadfile()

# Streamlit Credentials
from PIL import Image
img = Image.open("walmart_logo.png")
st.image(img,width=700,use_column_width=True)
st.title("M5 Forecasting - Accuracy") # Title
st.header("Forecasting of sales of products at Walmart")

activities = ["State","Category","Evaluation"]
choice = st.sidebar.selectbox("Select by",activities)

if choice == "State":
	st.subheader("Time Series Forecasting by State and Store")
	
	store_list = tuple()
	state = st.multiselect("Select State:",("CA","TX","WI"))
	if "CA" in state:
		store_list += ("CA_1","CA_2","CA_3","CA_4")
	if "TX" in state:
		store_list += ("TX_1","TX_2","TX_3")
	if "WI" in state:
		store_list += ("WI_1","WI_2","WI_3")
	store_id = st.multiselect("Select Store:",store_list)

	dept_list = tuple()
	category = st.multiselect("Select Category:",("HOBBIES","FOODS","HOUSEHOLD"))
	if "HOBBIES" in category:
		dept_list += ("HOBBIES_1","HOBBIES_2")
	if "FOODS" in category:
		dept_list += ("FOODS_1","FOODS_2","FOODS_3")
	if "HOUSEHOLD" in category:
		dept_list += ("HOUSEHOLD_1","HOUSEHOLD_2")
	dept_id = st.multiselect("Select Department:",dept_list)

	item_list = []
	for index,row in sales.iterrows():
		if len(state) == 0:
			item_list.append(row['id'])
		if len(state) > 0:
			if len(store_id) == 0:
				if len(category) == 0:
					if len(dept_id) == 0:
						if row['state_id'] in state:
							item_list.append(row['id'])
				if len(category) > 0:
					if len(dept_id) == 0:
						if row['state_id'] in state:
							if row['cat_id'] in category:
								item_list.append(row['id'])
					if len(dept_id) > 0:
							if row['state_id'] in state:
								if row['cat_id'] in category:
									if row['dept_id'] in dept_id:
										item_list.append(row['id'])

			if len(store_id) > 0:
				if len(category) == 0:
					if len(dept_id) == 0:
						if row['store_id'] in store_id:
							item_list.append(row['id'])
				if len(category) > 0:
					if len(dept_id) == 0:
						if row['store_id'] in store_id:
							if row['cat_id'] in category:
								item_list.append(row['id'])
					if len(dept_id) > 0:
						if row['store_id'] in store_id:
							if row['cat_id'] in category:
								if row['dept_id'] in dept_id:
									item_list.append(row['id'])
		
	item_id = st.multiselect("Select Item (Select upto 5 items):",item_list)

	st.write("Select date range for training and validation sets between 1/29/2011 and 4/24/2016")
	lower_train = str(st.date_input("Enter lower training set date",datetime.date(2011,1,29)))
	upper_train = str(st.date_input("Enter upper training set date",datetime.date(2016,3,26)))
	st.write("Validation period should be exactly equal to 30 days")
	lower_val = str(st.date_input("Enter lower validation date",datetime.date(2016,3,26)))
	upper_val = str(st.date_input("Enter upper validation date",datetime.date(2016,4,24)))
	algo_selected = st.radio("Select Algorithm for forecasting:",("Moving Average","ARIMA"))
	date_range = pd.date_range(start='1/29/2011', end='4/24/2016').astype(str)

	if st.button("SUBMIT"):
		if len(state) > 0:
			st.success("Submitted Successfully!")
			if len(item_id) == 0:
				df = sales.loc[sales['id'] == 'HOUSEHOLD_2_090_CA_4_validation']
				for value in item_list:
				    df = df.append(sales.loc[sales['id'] == value])
				df.drop(10333, axis=0, inplace = True)
				df = df.reset_index(drop=True).loc[:,'d_1':]
				for i in range(1,1914):
					df.rename(columns={'d_'+str(i):date_range[i-1]},inplace=True)
				df = df.append(pd.Series(df.mean(axis=0), index=df.columns), ignore_index=True)
				mean_index = (df.shape[0])-1

				train_dataset = df.loc[:,lower_train:upper_train]
				val_dataset = df.loc[:,lower_val:upper_val]

				if algo_selected == "Moving Average":
					predictions = []
					for i in range(len(val_dataset.columns)):
						if i == 0:
							predictions.append(np.mean(train_dataset[train_dataset.columns[-30:]].values, axis=1))
						if i < 31 and i > 0:
							predictions.append(0.5 * (np.mean(train_dataset[train_dataset.columns[-30+i:]].values, axis=1)+np.mean(predictions[:i], axis=0)))
						if i > 31:
							predictions.append(np.mean([predictions[:i]], axis=1))
					predictions = np.transpose(np.array([row.tolist() for row in predictions]))
					error_avg = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])

				if algo_selected == "ARIMA":
					predictions = []
					for row in tqdm.tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
						fit = sm.tsa.statespace.SARIMAX(row, seasonal_order=(0, 1, 1, 7)).fit()
						predictions.append(fit.forecast(30))
					predictions = np.array(predictions).reshape((-1, 30))
					error_arima = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])

				# if algo_selected == "Prophet":
				# 	dates = ["2007-12-" + str(i) for i in range(1, 31)]
				# 	predictions = []
				# 	for row in tqdm.tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
				# 		df_prophet = pd.DataFrame(np.transpose([dates, row]))
				# 		df_prophet.columns = ["ds", "y"]
				# 		model = Prophet(daily_seasonality=True)
				# 		model.fit(df_prophet)
				# 		future = model.make_future_dataframe(periods=30)
				# 		forecast = model.predict(future)["yhat"].loc[30:].values
				# 		predictions.append(forecast)
				# 	predictions = np.array(predictions).reshape((-1, 30))
				# 	error_prophet = np.linalg.norm(predictions[:len(item_id)] - val_dataset.values[:len(item_id)])/len(predictions[0])

				fig = make_subplots(rows=1, cols=1)
				fig.add_trace(go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[mean_index].values, marker=dict(color="red"),name="Trained signal"),row=1, col=1)
				fig.add_trace(go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[mean_index].values, mode='lines', marker=dict(color="green"),name="Validated signal"),row=1, col=1)
				fig.add_trace(go.Scatter(x=np.arange(70, 100), y=predictions[-1], mode='lines', marker=dict(color="black"),name="Predicted Signal"),row=1, col=1)
				fig.show()
				st.pyplot()

			if len(item_id) > 0:
				df = sales.loc[sales['id'] == 'HOUSEHOLD_2_090_CA_4_validation']
				for value in item_list:
				    df = df.append(sales.loc[sales['id'] == value])
				df.drop(10333, axis=0, inplace = True)
				df = df.reset_index(drop=True).loc[:,'d_1':]
				for i in range(1,1914):
					df.rename(columns={'d_'+str(i):date_range[i-1]},inplace=True)
				df = df.append(pd.Series(df.mean(axis=0), index=df.columns), ignore_index=True)
				mean_index = (df.shape[0])-1

				train_dataset = df.loc[:,lower_train:upper_train]
				val_dataset = df.loc[:,lower_val:upper_val]  

				if algo_selected == "Moving Average":
					predictions = []
					for i in range(len(val_dataset.columns)):
						if i == 0:
							predictions.append(np.mean(train_dataset[train_dataset.columns[-30:]].values, axis=1))
						if i < 31 and i > 0:
							predictions.append(0.5 * (np.mean(train_dataset[train_dataset.columns[-30+i:]].values, axis=1)+np.mean(predictions[:i], axis=0)))
						if i > 31:
							predictions.append(np.mean([predictions[:i]], axis=1))
					predictions = np.transpose(np.array([row.tolist() for row in predictions]))
					error_avg = np.linalg.norm(predictions[:len(item_id)] - val_dataset.values[:len(item_id)])/len(predictions[0])

				if algo_selected == "ARIMA":
					predictions = []
					for row in tqdm.tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
						fit = sm.tsa.statespace.SARIMAX(row, seasonal_order=(0, 1, 1, 7)).fit()
						predictions.append(fit.forecast(30))
					predictions = np.array(predictions).reshape((-1, 30))
					error_arima = np.linalg.norm(predictions[:len(item_id)] - val_dataset.values[:len(item_id)])/len(predictions[0])

				# if algo_selected == "Prophet":
				# 	dates = ["2007-12-" + str(i) for i in range(1, 31)]
				# 	predictions = []
				# 	for row in tqdm.tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
				# 		df_prophet = pd.DataFrame(np.transpose([dates, row]))
				# 		df_prophet.columns = ["ds", "y"]
				# 		model = Prophet(daily_seasonality=True)
				# 		model.fit(df_prophet)
				# 		future = model.make_future_dataframe(periods=30)
				# 		forecast = model.predict(future)["yhat"].loc[30:].values
				# 		predictions.append(forecast)
				# 	predictions = np.array(predictions).reshape((-1, 30))
				# 	error_prophet = np.linalg.norm(predictions[:len(item_id)] - val_dataset.values[:len(item_id)])/len(predictions[0])

				fig = make_subplots(rows=len(item_id), cols=1)
				for i in range(len(item_id)):
					fig.add_trace(go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[i].values, marker=dict(color="red"),name="Trained signal"),row=i+1, col=1)
					fig.add_trace(go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[i].values, mode='lines', marker=dict(color="green"), name="Validated signal"),row=i+1, col=1)
					fig.add_trace(go.Scatter(x=np.arange(70, 100), y=predictions[i], mode='lines', marker=dict(color="black"),name="Predicted Signal"),row=i+1, col=1)
				fig.show()
				st.pyplot()			
		else:
			st.error("Input not specified")

if choice == "Category":
	st.subheader("Time Series Forecasting by Category and Department")

	dept_list = tuple()
	category = st.multiselect("Select Category:",("HOBBIES","FOODS","HOUSEHOLD"))
	if "HOBBIES" in category:
		dept_list += ("HOBBIES_1","HOBBIES_2")
	if "FOODS" in category:
		dept_list += ("FOODS_1","FOODS_2","FOODS_3")
	if "HOUSEHOLD" in category:
		dept_list += ("HOUSEHOLD_1","HOUSEHOLD_2")
	dept_id = st.multiselect("Select Department:",dept_list)

	store_list = tuple()
	state = st.multiselect("Select State:",("CA","TX","WI"))
	if "CA" in state:
		store_list += ("CA_1","CA_2","CA_3","CA_4")
	if "TX" in state:
		store_list += ("TX_1","TX_2","TX_3")
	if "WI" in state:
		store_list += ("WI_1","WI_2","WI_3")
	store_id = st.multiselect("Select Store:",store_list)

	item_list = []
	for index,row in sales.iterrows():
		if len(category) == 0:
			item_list.append(row['id'])
		if len(category) > 0:
			if len(dept_id) == 0:
				if len(state) == 0:
					if len(store_id) == 0:
						if row['cat_id'] in category:
							item_list.append(row['id'])
				if len(state) > 0:
					if len(store_id) == 0:
						if row['cat_id'] in category:
							if row['store_id'] in store_id:
								item_list.append(row['id'])
					if len(store_id) > 0:
							if row['cat_id'] in category:
								if row['state_id'] in state:
									if row['store_id'] in store_id:
										item_list.append(row['id'])

			if len(dept_id) > 0:
				if len(state) == 0:
					if len(store_id) == 0:
						if row['dept_id'] in dept_id:
							item_list.append(row['id'])
				if len(state) > 0:
					if len(store_id) == 0:
						if row['dept_id'] in dept_id:
							if row['state_id'] in state:
								item_list.append(row['id'])
					if len(store_id) > 0:
						if row['dept_id'] in dept_id:
							if row['state_id'] in state:
								if row['store_id'] in store_id:
									item_list.append(row['id'])

	item_id = st.multiselect("Select Item (Select upto 5 items):",item_list)

	st.write("Select date range for training and validation sets between 1/29/2011 and 4/24/2016")
	lower_train = str(st.date_input("Enter lower training set date",datetime.date(2011,1,29)))
	upper_train = str(st.date_input("Enter upper training set date",datetime.date(2016,3,26)))
	st.write("Validation period should be exactly equal to 30 days")
	lower_val = str(st.date_input("Enter lower validation date",datetime.date(2016,3,26)))
	upper_val = str(st.date_input("Enter upper validation date",datetime.date(2016,4,24)))
	algo_selected = st.radio("Select Algorithm for forecasting:",("Moving Average","ARIMA","Prophet"))
	date_range = pd.date_range(start='1/29/2011', end='4/24/2016').astype(str)

	if st.button("SUBMIT"):
		if len(category) > 0:
			st.success("Submitted Successfully!")
			if len(item_id) == 0:
				df = sales.loc[sales['id'] == 'HOUSEHOLD_2_090_CA_4_validation']
				for value in item_list:
				    df = df.append(sales.loc[sales['id'] == value])
				df.drop(10333, axis=0, inplace = True)
				df = df.reset_index(drop=True).loc[:,'d_1':]
				for i in range(1,1914):
					df.rename(columns={'d_'+str(i):date_range[i-1]},inplace=True)
				df = df.append(pd.Series(df.mean(axis=0), index=df.columns), ignore_index=True)
				mean_index = (df.shape[0])-1

				train_dataset = df.loc[:,lower_train:upper_train]
				val_dataset = df.loc[:,lower_val:upper_val]

				if algo_selected == "Moving Average":
					predictions = []
					for i in range(len(val_dataset.columns)):
						if i == 0:
							predictions.append(np.mean(train_dataset[train_dataset.columns[-30:]].values, axis=1))
						if i < 31 and i > 0:
							predictions.append(0.5 * (np.mean(train_dataset[train_dataset.columns[-30+i:]].values, axis=1)+np.mean(predictions[:i], axis=0)))
						if i > 31:
							predictions.append(np.mean([predictions[:i]], axis=1))
					predictions = np.transpose(np.array([row.tolist() for row in predictions]))
					error_avg = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])

				if algo_selected == "ARIMA":
					predictions = []
					for row in tqdm.tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
						fit = sm.tsa.statespace.SARIMAX(row, seasonal_order=(0, 1, 1, 7)).fit()
						predictions.append(fit.forecast(30))
					predictions = np.array(predictions).reshape((-1, 30))
					error_arima = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])

				# if algo_selected == "Prophet":
				# 	dates = ["2007-12-" + str(i) for i in range(1, 31)]
				# 	predictions = []
				# 	for row in tqdm.tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
				# 		df_prophet = pd.DataFrame(np.transpose([dates, row]))
				# 		df_prophet.columns = ["ds", "y"]
				# 		model = Prophet(daily_seasonality=True)
				# 		model.fit(df_prophet)
				# 		future = model.make_future_dataframe(periods=30)
				# 		forecast = model.predict(future)["yhat"].loc[30:].values
				# 		predictions.append(forecast)
				# 	predictions = np.array(predictions).reshape((-1, 30))
				# 	error_prophet = np.linalg.norm(predictions[:len(item_id)] - val_dataset.values[:len(item_id)])/len(predictions[0])

				fig = make_subplots(rows=1, cols=1)
				fig.add_trace(go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[mean_index].values, marker=dict(color="red"),name="Trained signal"),row=1, col=1)
				fig.add_trace(go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[mean_index].values, mode='lines', marker=dict(color="green"),name="Validated signal"),row=1, col=1)
				fig.add_trace(go.Scatter(x=np.arange(70, 100), y=predictions[-1], mode='lines', marker=dict(color="black"),name="Predicted Signal"),row=1, col=1)
				fig.show()
				st.pyplot()

			if len(item_id) > 0:
				df = sales.loc[sales['id'] == 'HOUSEHOLD_2_090_CA_4_validation']
				for value in item_list:
				    df = df.append(sales.loc[sales['id'] == value])
				df.drop(10333, axis=0, inplace = True)
				df = df.reset_index(drop=True).loc[:,'d_1':]
				for i in range(1,1914):
					df.rename(columns={'d_'+str(i):date_range[i-1]},inplace=True)
				df = df.append(pd.Series(df.mean(axis=0), index=df.columns), ignore_index=True)
				mean_index = (df.shape[0])-1

				train_dataset = df.loc[:,lower_train:upper_train]
				val_dataset = df.loc[:,lower_val:upper_val]  

				if algo_selected == "Moving Average":
					predictions = []
					for i in range(len(val_dataset.columns)):
						if i == 0:
							predictions.append(np.mean(train_dataset[train_dataset.columns[-30:]].values, axis=1))
						if i < 31 and i > 0:
							predictions.append(0.5 * (np.mean(train_dataset[train_dataset.columns[-30+i:]].values, axis=1)+np.mean(predictions[:i], axis=0)))
						if i > 31:
							predictions.append(np.mean([predictions[:i]], axis=1))
					predictions = np.transpose(np.array([row.tolist() for row in predictions]))
					error_avg = np.linalg.norm(predictions[:len(item_id)] - val_dataset.values[:len(item_id)])/len(predictions[0])

				if algo_selected == "ARIMA":
					predictions = []
					for row in tqdm.tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
						fit = sm.tsa.statespace.SARIMAX(row, seasonal_order=(0, 1, 1, 7)).fit()
						predictions.append(fit.forecast(30))
					predictions = np.array(predictions).reshape((-1, 30))
					error_arima = np.linalg.norm(predictions[:len(item_id)] - val_dataset.values[:len(item_id)])/len(predictions[0])

				# if algo_selected == "Prophet":
				# 	dates = ["2007-12-" + str(i) for i in range(1, 31)]
				# 	predictions = []
				# 	for row in tqdm.tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
				# 		df_prophet = pd.DataFrame(np.transpose([dates, row]))
				# 		df_prophet.columns = ["ds", "y"]
				# 		model = Prophet(daily_seasonality=True)
				# 		model.fit(df_prophet)
				# 		future = model.make_future_dataframe(periods=30)
				# 		forecast = model.predict(future)["yhat"].loc[30:].values
				# 		predictions.append(forecast)
				# 	predictions = np.array(predictions).reshape((-1, 30))
				# 	error_prophet = np.linalg.norm(predictions[:len(item_id)] - val_dataset.values[:len(item_id)])/len(predictions[0])

				fig = make_subplots(rows=len(item_id), cols=1)
				for i in range(len(item_id)):
					fig.add_trace(go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[i].values, marker=dict(color="red"),name="Trained signal"),row=i+1, col=1)
					fig.add_trace(go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[i].values, mode='lines', marker=dict(color="green"), name="Validated signal"),row=i+1, col=1)
					fig.add_trace(go.Scatter(x=np.arange(70, 100), y=predictions[i], mode='lines', marker=dict(color="black"),name="Predicted Signal"),row=i+1, col=1)
				fig.show()
				st.pyplot()			
		else:
			st.error("Input not specified")

if choice == "Evaluation":
	st.subheader("Evaluation Report")

	if st.button("Show Report"):
		d_cols = [c for c in sales.columns if 'd_' in c]
		train_dataset = sales[d_cols[-100:-30]]
		val_dataset = sales[d_cols[-30:]]

		predictions = []
		for i in range(len(val_dataset.columns)):
			if i == 0:
				predictions.append(np.mean(train_dataset[train_dataset.columns[-30:]].values, axis=1))
			if i < 31 and i > 0:
				predictions.append(0.5 * (np.mean(train_dataset[train_dataset.columns[-30+i:]].values, axis=1)+np.mean(predictions[:i], axis=0)))
			if i > 31:
				predictions.append(np.mean([predictions[:i]], axis=1))
		predictions = np.transpose(np.array([row.tolist() for row in predictions]))
		error_avg = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])


		predictions = []
		for row in tqdm.tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
			fit = sm.tsa.statespace.SARIMAX(row, seasonal_order=(0, 1, 1, 7)).fit()
			predictions.append(fit.forecast(30))
		predictions = np.array(predictions).reshape((-1, 30))
		error_arima = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])


		dates = ["2007-12-" + str(i) for i in range(1, 31)]
		predictions = []
		for row in tqdm.tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
			df_prophet = pd.DataFrame(np.transpose([dates, row]))
			df_prophet.columns = ["ds", "y"]
			model = Prophet(daily_seasonality=True)
			model.fit(df_prophet)
			future = model.make_future_dataframe(periods=30)
			forecast = model.predict(future)["yhat"].loc[30:].values
			predictions.append(forecast)
		predictions = np.array(predictions).reshape((-1, 30))
		error_prophet = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])

		# days = range(1, 1913 + 1)
		# time_series_columns = [f'd_{i}' for i in days]
		# time_series_data = sales[time_series_columns]
		# forecast = pd.DataFrame(time_series_data.iloc[:, -28:].mean(axis=1))
		# forecast = pd.concat([forecast] * 28, axis=1)
		# forecast.columns = [f'F{i}' for i in range(1, forecast.shape[1] + 1)]
		# validation_ids = sales['id'].values
		# evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]
		# ids = np.concatenate([validation_ids, evaluation_ids])
		# predictions = pd.DataFrame(ids, columns=['id'])
		# forecast = pd.concat([forecast] * 2).reset_index(drop=True)
		# predictions = pd.concat([predictions, forecast], axis=1)
		# predictions.to_csv('submission.csv', index=False)

		error = [error_avg, error_arima]
		names = ["Moving average", "ARIMA"]
		df = pd.DataFrame(np.transpose([error, names]))
		df.columns = ["RMSE Loss", "Model"]
		fig7 = px.bar(df, y="RMSE Loss", x="Model", color="Model", title="RMSE Loss vs. Model")
		fig7.show()
		st.pyplot()


