import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/decision_model.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('/content/drive/My Drive/harshita PIET18CS061 - PCA and NN Dataset11.csv')
# Extracting independent variable:
X = dataset.iloc[:,0:10].values
# Encoding the Independent Variable
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'constant', fill_value="Female", verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,3:4]) 
#Replacing missing data with the calculated mean value  
X[:,3:4]= imputer.transform(X[:, 3:4])


imputer = SimpleImputer(missing_values= np.NaN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,4:10]) 
#Replacing missing data with the calculated mean value  
X[:,4:10]= imputer.transform(X[:,4:10]) 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
labelencoder_X = LabelEncoder()
X[:,2] = labelencoder_X.fit_transform(X[:,2])
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict(Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary):
  output= model.predict(sc.transform([[Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary]]))
  print("Exited", output)
  if output==[0]:
    print("Will not exit")
  elif output==[1]:
    print("Will Exit")
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Customer will exit or not using Decision Tree Classification")
    Age = st.text_input("age","")
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    CreditScore = st.number_input('Insert CreditScore')
 
    Surname = st.number_input('Insert Surname')
    Tenure  = st.number_input('Insert Tenure ')
    Balance = st.number_input('Insert Balance')
    
    HasCrCard=	 st.number_input('Insert HasCrCard ')
    thalach =	 st.number_input('Insert thalach')
    IsActiveMember=	 st.number_input('Insert IsActiveMember')
    EstimatedSalary =	 st.number_input('Insert EstimatedSalary ')
    
    Geography=st.number_input('Insert Gender France:1 Spain:0')
    Gender = st.number_input('Insert Gender Male:1 Female:0')
    
   
  
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Harshita pal")
      st.subheader("Head , Department of Computer Engineering")

if __name__=='__main__':
  main()