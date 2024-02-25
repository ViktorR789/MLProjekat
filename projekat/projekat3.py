from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

carTrain = pd.read_csv('C:\\Users\\pcpc\\Desktop\\projekat\\Train_Dataset.csv')
carTest = pd.read_csv('C:\\Users\\pcpc\\Desktop\\projekat\\Test_Dataset.csv')



yTrain = carTrain['Default']
print(yTrain.value_counts())
carTrain.drop(columns=['Default'], inplace=True)


for col in carTrain.columns:
    if carTrain[col].dtype in ['int64', 'float64']:
        imputer = SimpleImputer(strategy='mean')
    else:
        imputer = SimpleImputer(strategy='most_frequent')
    carTrain.loc[:,col] = imputer.fit_transform(carTrain[[col]])[:,0]
    #print(imputer.fit_transform(carTrain[[col]]).shape)
    carTest.loc[:,col] = imputer.transform(carTest[[col]])[:, 0]
    
    

numericalColumns = carTrain.select_dtypes(include=['int64', 'float64'])
corrMatrix = numericalColumns.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

carTrain['Children_Percentage'] = carTrain['Child_Count'] / carTrain['Client_Family_Members']
carTest['Children_Percentage'] = carTest['Child_Count'] / carTest['Client_Family_Members']
carTrain.drop(columns=['Client_Family_Members', 'Child_Count'], inplace=True)
carTest.drop(columns=['Client_Family_Members', 'Child_Count'], inplace=True)
carTrain.drop(columns=['Loan_Annuity'], inplace=True)
carTest.drop(columns=['Loan_Annuity'], inplace=True)


oneHot = ["Accompany_Client", "Client_Income_Type", "Client_Gender", "Client_Housing_Type", "Client_Occupation"]
combinedData = pd.concat([carTrain, carTest])
combinedData = pd.get_dummies(combinedData, columns=oneHot)

carTrain = combinedData[:len(carTrain)]
carTest = combinedData[len(carTrain):]

labelCols = ["Client_Education", "Client_Marital_Status", "Loan_Contract_Type", 
              "Client_Permanent_Match_Tag", "Client_Contact_Work_Tag","Type_Organization"]
for col in labelCols:
    le = LabelEncoder()
    carTrain.loc[:,col] = le.fit_transform(carTrain[col])
    carTest.loc[:,col] = le.transform(carTest[col])

scaler = StandardScaler()
scaledTrain = scaler.fit_transform(carTrain)
scaledTest = scaler.transform(carTest)

#print(scaledTrain[:5])

pca = PCA(n_components=0.95)
pcaTrain= pca.fit_transform(scaledTrain)
pcaTest = pca.transform(scaledTest)

undersampler = RandomUnderSampler(sampling_strategy={0: 9800, 1: 1800}, random_state=7, replacement=False)
xtrResample, YtrResample = undersampler.fit_resample(pcaTrain, yTrain)

rfClassifier = RandomForestClassifier(n_estimators=5, random_state=20)
rfClassifier.fit(xtrResample, YtrResample)

yTrainPred = rfClassifier.predict(xtrResample)

predTest = rfClassifier.predict(pcaTest)

predvidjanjaCSV = pd.DataFrame({'ClientId': carTest['ID'].astype(int), 'Default': predTest.astype(int)})
predvidjanjaCSV.to_csv('C:\\Users\\pcpc\\Desktop\\projekat\\resenje.csv', index=0)

accuracy = accuracy_score(YtrResample, yTrainPred)
print("Accuracy:", accuracy)

precision = precision_score(YtrResample, yTrainPred)
print("Precision:", precision)

recall = recall_score(YtrResample, yTrainPred)
print("Recall:", recall)

f1 = f1_score(YtrResample, yTrainPred)
print("F1-score:", f1)