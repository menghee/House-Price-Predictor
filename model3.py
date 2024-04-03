import pandas as pd 
from sklearn.linear_model import LinearRegression
import pickle
df=pd.read_json('Data/Salmon_Price2.json')
x=df['Length(in cm)'].values.reshape(-1,1)
y=df['Price(in $)'].values.reshape(-1,1)
lin=LinearRegression()
lin.fit(x,y)
pickle.dump(lin,open('model3.pkl','wb'))