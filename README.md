# Car_Price_Prediction
## Overview
This project aims at predicting the resale value of used vehicles using a regression model based on Random Forest Regressor algorithm. This model takes 8 parameters like car_name, fuel_type, Kms_driven, etc of a pre-owned car as input and tries to predict its selling_price.

## Motivation
The rise of taxi aggregators (like Ola, Uber) and rental services has changed people's mindset towards owning a full-time car. On top of this, government's push towards a circular economy and emergence of new Hybrid / Electric vehicles has resulted in surge in sale of pre-owned vehicles. And hence, the need for a machine learning model that helps customers in estimating sale price of their vehicles.

## Steps

### Exploratory Data Analysis
EDA was done to understand the relationship between independent variables amongst themselves as well as with dependent variable (selling price).

 ![image](https://user-images.githubusercontent.com/86396532/123532673-d1017a80-d72c-11eb-80fe-9684995b12e5.png)
 
Aboce graph is a pair plot that shows features relationship amongst themselves  
 
 ![image](https://user-images.githubusercontent.com/86396532/123307091-b322f280-d53f-11eb-81bb-f978fc995241.png)
 
 The above histogram reflects that the sale of vehicles fell post 2015. It can be associated with the fact that people who were planning to buy new vehicle have been holding their decision due to gradual entry of electric vehicles and government's evolving regulations. The graph is left skewed and hence doesn't follow a gaussian distribution. If we had been using ML algorithms like linear regression, standardisation of this data would have been required 
 
 ![image](https://user-images.githubusercontent.com/86396532/123307774-81f6f200-d540-11eb-9ca4-63c94c73128c.png)

Bar chart shows that Indian vehicle market is still dominated by petrol vehicles, followed by diesel and CNG. Also the entry of automatic transmission cars and its overall acceptance amongst customers have been slow which also gets displayed.

![image](https://user-images.githubusercontent.com/86396532/123308229-0b0e2900-d541-11eb-8fbc-dd9563763d27.png)

Above graph shows that sale of pre-owned cars through dealers are almost twice that of individuals . Dealers often act as channels that connect the buyer and the seller. But the rise of social media platforms (eg FB) has made this middle man insignificant.

![image](https://user-images.githubusercontent.com/86396532/123308720-a2737c00-d541-11eb-84d4-59aea23d6e9b.png)

The density function shows that majority of the cars that were sold in secondary market were driven for around 50,000km or less. But further down we will see that Kms_driven hardly has any relationship with the Selling_price of the car and hence would be dropped while model training.

![image](https://user-images.githubusercontent.com/86396532/123308933-ec5c6200-d541-11eb-9776-0b64b9e1c71b.png)

Resale price of vehicles are almost directly proportionate to Current prices which is pretty intuitive.

![image](https://user-images.githubusercontent.com/86396532/123309034-0b5af400-d542-11eb-9620-f1767abaf8d4.png)

This heatmap quantifies the correlation that different features have amongst themselves. It shows that current prices and Year have significant impact over selling price but surprisingly distance driven (Kms_driven) hardly has any relationship with it.

### Encoding
To inculcate the impact of categorical variables (Fuel_type, Transmission, etc) on the dependent variable (selling_price) in our machine learning model, it is important to convert them to some numerical values. So One Hot Encoding was used for Fuel_type, Transmission and Seller_type variables whereas Count/Frequency encoding was used for Car_Name (One Hot encoding would have resulted in curse of dimensionality due to high variety of classes).

### Model Creation
For this regression problem, I have used Random Forest Regressor algorithm to create a machine learning model. Since its an ensemble technique, it doesn't require any scaling or normalizing of data. Training data and Test data was split in the ratio (4:1) using the Train_test_split library. While creating the training dataset 'Kms_Driven' feature was dropped as it barely had any correlation with selling_price (observed in heatmap). This helped reducing complexity and computational cost of algorithm as well as slightly reducing the prediction error.

### Testing
The model performed well on the test dataset with R_square value being close to 0.90 whereas Mean_Square_Error (MSE) being around 1.0

## Conclusion
Though the algorithm shows low error, probably due to inherent strength of ensemble techniques, the reliability of this model stays questionable due to lack of sufficient amount of input data. The dataset had mere 301 entries which is not enough for a model to become effective through training.
