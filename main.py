import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

#for offline plotting
from plotly.offline import download_plotlyjs,init_notebook_mode, plot , iplot
init_notebook_mode(connected=True)

DOC = pd.read_csv("C:/stock market prediction dataset.csv")
print (DOC)

DOC.info()
DOC['Date']  = pd.to_datetime(DOC['Date'])
print (f'DataFrame Contains stock prices between  {DOC.Date.min()}   {DOC.Date.max()}')
print (f'TOTAL DAYS = {( DOC.Date.max()   -   DOC.Date.min() ).days} days')
DOC.describe()

DOC [[ "Open", "High","Low","Close","Last","VWAP"]].plot(kind= "box")
#setting the layout for our plot
layout = go.Layout(
        title= 'STOCK PRICES OF COMPANY',
        xaxis=dict(
              title = 'Date',
              titlefont=dict(
                    family= "Courier New,monospace",
                    size=18,
                    color="#7f7f7f"

              )
        ),
       yaxis=dict(
               title = "Price",
               titlefont=dict(
                     family="Courier New,monospace",
                     size = 18,
                     color = '#7f7f7f'

      ) )
)
plt.show()

DOC_data = [{"x" : DOC["Date"],  "y" : DOC["Close"]}]
plot = go.Figure(data=DOC_data, layout=layout)




#BUILDING THE REGRESSION MODEL
from sklearn.model_selection import train_test_split

#FOR PREPROCESSING
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#FOR model evalution
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


#spilt the data into train and test sets
X = np.array(DOC.index).reshape(-1,1)
Y = DOC['Close']
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3, random_state=101)

#feature scaling
scaler= StandardScaler().fit(X_train)
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

#CREATING A LINEAR MODEL
Lm= LinearRegression()
Lm.fit (X_train, Y_train)

#plot actual and predicted values for train dataset
trace0 = go.Scatter(
    x = X_train.T[0],
    y = Y_train,
    mode = "markers",
    name = "Actual"
)

trace1 = go.Scatter(
        x = X_train.T[0],
        y = Lm.predict(X_train).T,
        mode = 'lines',
        name = "Predict"
)
plt.show()
DOC_data = [trace0,trace1]
layout.xaxis.title.text = "Day"
plot2 = go.Figure(data = DOC_data, layout = layout)

# CALCULATE SCORESFOR MODEL EVOLUTION
SCORES = ''' '''
{'METRIC'.ljust(10)}
{'TRAIN'.center(20)}
{'TEST'.center(20)}
{'r2_score'.ljust(10)}
{r2_score(Y_train,  Lm.predict ( X_train))}
{r2_score(Y_test, Lm.predict(X_test))}
{'MSE'.ljust(10)}
{mse(Y_train, Lm.predict(X_train))}
{mse(Y_test, Lm.predict(X_test))}
''' '''
print (SCORES)
