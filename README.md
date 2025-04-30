# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AMIRTHAVARSHINI V
RegisterNumber:  212223040014
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset.info()
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary', axis=1)
dataset.info()
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
dataset.info()
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()
print(x_train.shape)
print(y_train.shape)
from sklearn.linear_model import LogisticRegression
cl=LogisticRegression(max_iter=1000)
cl.fit(x_train,y_train)
y_pred=cl.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_pred,y_test))
confusion_matrix(y_pred,y_test)
cl.predict([[0,87,0,95,0,2,8,0,0,1,5,6]])
cl.predict([[1,2,3,4,5,6,7,8,9,10,11,12]])
```
## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![Screenshot 2025-04-29 223514](https://github.com/user-attachments/assets/2c6ab15d-6fcf-400d-aefe-40ab01358e05)
![Screenshot 2025-04-29 223524](https://github.com/user-attachments/assets/09b6f399-81cb-4c6f-9589-78027d3ef23a)
![Screenshot 2025-04-29 223538](https://github.com/user-attachments/assets/122ab87d-7219-4e71-9f3c-242c0c6ae835)
![Screenshot 2025-04-29 223548](https://github.com/user-attachments/assets/ec03e1b8-b34a-4ea3-9bc3-7fce42727a18)
![Screenshot 2025-04-29 223557](https://github.com/user-attachments/assets/7328b950-338a-4402-be19-5d149f65de19)
![Screenshot 2025-04-29 223825](https://github.com/user-attachments/assets/d52ca6bb-56b4-48ef-8309-6ebf98889086)
![Screenshot 2025-04-29 223839](https://github.com/user-attachments/assets/4c211e42-1908-4e4b-b1b8-4b04e64e5533)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
