import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
# Loading the Model
loaded_model=pickle.load(open('D:/MLProject/trained_model.sav','rb'))
# Load the scaler
scaler = pickle.load(open('D:/MLProject/scaler.sav', 'rb'))

input_data=(63,	1,	3	,145	,233	,1,	0	,150	,0	,2.3	,0	,0	,1)
input_data_as_ndarray=np.asarray(input_data)
input_data_reshaped=input_data_as_ndarray.reshape(1,-1)
scaler=MinMaxScaler()
input_data_scaled = scaler.transform(input_data_reshaped)
prediction=loaded_model.predict(input_data_scaled )
print(prediction)
if(prediction[0]==0):
 print('The Person does not have heart disease')
else:
 print('The Person has heart disease')