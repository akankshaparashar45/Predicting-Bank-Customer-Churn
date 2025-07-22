#!/usr/bin/env python
# coding: utf-8

# <center><font size=6> Bank Churn Prediction </font></center>

# ## Problem Statement

# ### Context

# Businesses like banks which provide service have to worry about problem of 'Customer Churn' i.e. customers leaving and joining another service provider. It is important to understand which aspects of the service influence a customer's decision in this regard. Management can concentrate efforts on improvement of service, keeping in mind these priorities.

# ### Objective

# You as a Data scientist with the  bank need to  build a neural network based classifier that can determine whether a customer will leave the bank  or not in the next 6 months.

# ### Data Dictionary

# * CustomerId: Unique ID which is assigned to each customer
# 
# * Surname: Last name of the customer
# 
# * CreditScore: It defines the credit history of the customer.
#   
# * Geography: A customer’s location
#    
# * Gender: It defines the Gender of the customer
#    
# * Age: Age of the customer
#     
# * Tenure: Number of years for which the customer has been with the bank
# 
# * NumOfProducts: refers to the number of products that a customer has purchased through the bank.
# 
# * Balance: Account balance
# 
# * HasCrCard: It is a categorical variable which decides whether the customer has credit card or not.
# 
# * EstimatedSalary: Estimated salary
# 
# * IsActiveMember: Is is a categorical variable which decides whether the customer is active member of the bank or not ( Active member in the sense, using bank products regularly, making transactions etc )
# 
# * Exited : whether or not the customer left the bank within six month. It can take two values
#     - 0 = No ( Customer did not leave the bank )
#     - 1 = Yes ( Customer left the bank )

# In[ ]:





# ## Importing necessary libraries

# # Installing the libraries with the specified version.
# !pip install tensorflow==2.15.0 scikit-learn==1.2.2 seaborn==0.13.1 matplotlib==3.7.1 numpy==1.25.2 pandas==2.0.3 imbalanced-learn==0.10.1 -q --user

# In[11]:


pip install imbalanced-learn


# **Note:** After running the above cell, please restart the notebook kernel/runtime (depending on whether you're using Jupyter Notebook or Google Colab) and then sequentially run all cells from the one below.

# In[1]:


# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Library to split data
from sklearn.model_selection import train_test_split

# library to import to standardize the data
from sklearn.preprocessing import StandardScaler, LabelEncoder

# importing different functions to build models
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout

# importing SMOTE
from imblearn.over_sampling import SMOTE

# importing metrics
from sklearn.metrics import confusion_matrix,roc_curve,classification_report,recall_score

import random

# Library to avoid the warnings
import warnings
warnings.filterwarnings("ignore")


# ## Loading the dataset

# In[19]:


ds = pd.read_csv("C:\\Users\\hp\\Documents\\Great Learning_Introduction to Neural Network\\bank-1.csv")    # complete the code to load the dataset


# ## Data Overview

# ### View the first and last 5 rows of the dataset.

# In[20]:


# let's view the first 5 rows of the data
ds.head(5)


# In[21]:


# let's view the last 5 rows of the data
ds.tail(5) 


# ### Understand the shape of the dataset

# In[22]:


# Checking the number of rows and columns in the training data
ds.shape 


# ### Check the data types of the columns for the dataset

# In[23]:


ds.info()


# ### Checking the Statistical Summary

# In[24]:


ds.describe().T


# ### Checking for Missing Values

# In[25]:


# let's check for missing values in the train data
ds.isnull().sum()


# ### Checking for unique values for each of the column

# In[26]:


ds.nunique()


# In[27]:


ds.columns


# In[28]:


#RowNumber , CustomerId and Surname are unique hence dropping it
ds = ds.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
ds.head()


# In[29]:


ds.info()


# ## Exploratory Data Analysis

# ### Univariate Analysis

# In[30]:


# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# In[31]:


# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# #### Observations on CreditScore

# In[32]:


histogram_boxplot(ds,'CreditScore')


# #### Observations on Age

# In[34]:


histogram_boxplot(ds,'Age')         


# #### Observations on Balance

# In[36]:


histogram_boxplot(ds,'Balance')       


# #### Observations on Estimated Salary

# In[37]:


histogram_boxplot(ds,'EstimatedSalary')          


# #### Observations on Exited

# In[38]:


labeled_barplot(ds, "Exited", perc=True)


# #### Observations on Geography

# In[39]:


labeled_barplot(ds,'Geography')              


# #### Observations on Gender

# In[40]:


labeled_barplot(ds,'Gender')               


# #### Observations on Tenure

# In[41]:


labeled_barplot(ds,'Tenure')               


# #### Observations on Number of Products

# In[42]:


labeled_barplot(ds,'NumOfProducts')               


# #### Observations on Has Credit Card

# In[43]:


labeled_barplot(ds,'HasCrCard')               


# #### Observations on Is Active Member

# In[44]:


labeled_barplot(ds,'IsActiveMember')               


# ### Bivariate Analysis

# In[45]:


# function to plot stacked bar chart


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left",
        frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# #### Correlation plot

# In[46]:


# defining the list of numerical columns
cols_list = ["CreditScore","Age","Tenure","Balance","EstimatedSalary"]


# In[47]:


plt.figure(figsize=(15, 7))
sns.heatmap(ds[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# #### Exited Vs Geography

# In[48]:


stacked_barplot(ds, "Geography", "Exited" )


# #### Exited Vs Gender

# In[49]:


stacked_barplot(ds, "Gender", "Exited" )                   


# #### Exited Vs Has Credit Card

# In[50]:


stacked_barplot(ds, "HasCrCard", "Exited" )                  


# #### Exited Vs Is active member

# In[51]:


stacked_barplot(ds, "IsActiveMember", "Exited")                   


# #### Exited Vs Credit Score

# In[52]:


plt.figure(figsize=(5,5))
sns.boxplot(y='CreditScore',x='Exited',data=ds)
plt.show()


# #### Exited Vs Age

# In[53]:


plt.figure(figsize=(5,5))
sns.boxplot(y='Age',x='Exited',data=ds)               
plt.show()


# #### Exited Vs Tenure

# In[54]:


plt.figure(figsize=(5,5))
sns.boxplot(y='Tenure',x='Exited',data=ds)               
plt.show()


# #### Exited Vs Balance

# In[55]:


plt.figure(figsize=(5,5))
sns.boxplot(y='Balance',x='Exited',data=ds)              
plt.show()


# #### Exited Vs Number of Products

# In[56]:


plt.figure(figsize=(5,5))
sns.boxplot(y='NumOfProducts',x='Exited',data=ds)               
plt.show()


# #### Exited Vs Estimated Salary

# In[57]:


plt.figure(figsize=(5,5))
sns.boxplot(y='EstimatedSalary',x='Exited',data=ds)               
plt.show()


# ## Data Preprocessing

# ### Dummy Variable Creation

# In[58]:


ds = pd.get_dummies(ds,columns=ds.select_dtypes(include=["object"]).columns.tolist(),drop_first=True,dtype=float)


# In[59]:


ds.info()


# In[62]:


ds.head(5)


# ### Train-validation-test Split

# In[63]:


X = ds.drop(['Exited'],axis=1) # Credit Score through Estimated Salary
y = ds['Exited'] # Exited


# In[64]:


# Splitting the dataset into the Training and Testing set.

X_large, X_test, y_large, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42,stratify=y,shuffle = True) ## Complete the code to Split the X and y and obtain test set


# In[65]:


# Splitting the dataset into the Training and Testing set.

X_train, X_val, y_train, y_val = train_test_split(X_large, y_large, test_size = 0.3, random_state = 42,stratify=y_large, shuffle = True) ## complete the code to Split X_large and y_large to obtain train and validation sets


# In[66]:


print(X_train.shape, X_val.shape, X_test.shape)


# In[67]:


print(y_train.shape, y_val.shape, y_test.shape)


# ### Data Normalization

# Since all the numerical values are on a different scale, so we will be scaling all the numerical values to bring them to the same scale.

# In[68]:


# creating an instance of the standard scaler
sc = StandardScaler()

X_train[cols_list] = sc.fit_transform(X_train[cols_list])
X_val[cols_list] = sc.transform(X_val[cols_list])    ## specify the columns to normalize
X_test[cols_list] = sc.transform(X_test[cols_list])    ## specify the columns to normalize


# ## Model Building

# ### Model Evaluation Criterion

# Write down the logic for choosing the metric that would be the best metric for this business scenario.
# 
# - The best metric is recall which is a metric used to evaluate a model's ability to correctly identify all relevant instances in a dataset.
# - It is defined as the ratio of True positive and True positive + False Negative
# - Recall should be prioritized when:
# * Missing a true positive has a high cost.
# * The dataset is imbalanced and you care about detecting rare events.
# * The focus is on maximizing coverage of positive instances, even at the expense of some false positives.

# **Let's create a function for plotting the confusion matrix**
# 
# 

# In[69]:


def make_confusion_matrix(actual_targets, predicted_targets):
    """
    To plot the confusion_matrix with percentages

    actual_targets: actual target (dependent) variable values
    predicted_targets: predicted target (dependent) variable values
    """
    cm = confusion_matrix(actual_targets, predicted_targets)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(cm.shape[0], cm.shape[1])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# Let's create two blank dataframes that will store the recall values for all the models we build.

# In[70]:


train_metric_df = pd.DataFrame(columns=["recall"])
valid_metric_df = pd.DataFrame(columns=["recall"])


# ### Neural Network with SGD Optimizer

# In[71]:


backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)


# In[72]:


#Initializing the neural network
model_0 = Sequential()
# Adding the input layer with 64 neurons and relu as activation function
model_0.add(Dense(64, activation='relu', input_dim = X_train.shape[1]))
# Add a hidden layer 
model_0.add(Dense(32, activation='relu'))
# Add the output layer with the number of neurons required
model_0.add(Dense(1, activation = 'sigmoid'))


# In[73]:


#Complete the code to use SGD as the optimizer.
optimizer = tf.keras.optimizers.SGD(0.001)

# uncomment one of the following lines to define the metric to be used
# metric = 'accuracy'
metric = keras.metrics.Recall()
# metric = keras.metrics.Precision()
# metric = keras.metrics.F1Score()


# In[74]:


## Compile the model with binary cross entropy as loss function and recall as the metric.
model_0.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=[metric])


# In[75]:


model_0.summary()


# In[94]:


# Fitting the ANN

history_0 = model_0.fit(
    X_train, y_train,
    batch_size=32,    ## specify the batch size to use
    validation_data=(X_val,y_val),
    epochs=850,    ## specify the number of epochs
    verbose=1
)


# **Loss function**

# In[77]:


#Plotting Train Loss vs Validation Loss
plt.plot(history_0.history['loss'])
plt.plot(history_0.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# **Recall**

# In[95]:


#Plotting Train recall vs Validation recall
plt.plot(history_0.history['recall'])
plt.plot(history_0.history['val_recall'])
plt.title('model recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[96]:


#Predicting the results using best as a threshold
y_train_pred = model_0.predict(X_train)
y_train_pred = (y_train_pred > 0.5)
y_train_pred


# In[97]:


#Predicting the results using best as a threshold
y_val_pred = model_0.predict(X_val)    ## make prediction on the validation set
y_val_pred = (y_val_pred > 0.5)
y_val_pred


# In[98]:


model_name = "NN with SGD"

train_metric_df.loc[model_name] = recall_score(y_train, y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val, y_val_pred)


# **Classification report**

# In[99]:


#lassification report
cr = classification_report(y_train, y_train_pred)
print(cr)


# In[100]:


#classification report
cr=classification_report(y_val, y_val_pred)    ## check the model's performance on the validation set
print(cr)


# **Confusion matrix**

# In[101]:


make_confusion_matrix(y_train, y_train_pred)


# In[102]:


make_confusion_matrix(y_val, y_val_pred)    ## check the model's performance on the validation set


# ## Model Performance Improvement

# ### Neural Network with Adam Optimizer

# In[103]:


backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)


# In[104]:


#Initializing the neural network
model_1 = Sequential()
#Add a input layer 
model_1.add(Dense(64,activation='relu',input_dim = X_train.shape[1]))
#Add a hidden layer 
model_1.add(Dense(32,activation='relu'))
#Add a output layer with the required number of neurons and an activation function
model_1.add(Dense(1, activation = 'sigmoid'))


# In[105]:


#Complete the code to use Adam as the optimizer.
optimizer = tf.keras.optimizers.Adam()

# uncomment one of the following lines to define the metric to be used
# metric = 'accuracy'
metric = keras.metrics.Recall()
# metric = keras.metrics.Precision()
# metric = keras.metrics.F1Score()


# In[106]:


# Compile the model with binary cross entropy as loss function and recall as the metric
model_1.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=[metric])


# In[107]:


model_1.summary()


# In[108]:


#Fitting the ANN
history_1 = model_1.fit(
    X_train,y_train,
    batch_size=32, ## specify the batch size to use
    validation_data=(X_val,y_val),
    epochs=150, ## specify the number of epochs
    verbose=1
)


# **Loss function**

# In[109]:


#Plotting Train Loss vs Validation Loss
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# **Recall**

# In[110]:


#Plotting Train recall vs Validation recall
plt.plot(history_1.history['recall'])
plt.plot(history_1.history['val_recall'])
plt.title('model recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[111]:


#Predicting the results using 0.5 as the threshold
y_train_pred = model_1.predict(X_train)
y_train_pred = (y_train_pred > 0.5)
y_train_pred


# In[112]:


#Predicting the results using 0.5 as the threshold
y_val_pred = model_1.predict(X_val)
y_val_pred = (y_val_pred > 0.5)
y_val_pred


# In[113]:


model_name = "NN with Adam"

train_metric_df.loc[model_name] = recall_score(y_train,y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val,y_val_pred)


# **Classification report**

# In[114]:


#lassification report
cr=classification_report(y_train,y_train_pred)
print(cr)


# In[115]:


#classification report
cr=classification_report(y_val,y_val_pred)  ## check the model's performance on the validation set
print(cr)


# **Confusion matrix**

# In[116]:


#Calculating the confusion matrix
make_confusion_matrix(y_train, y_train_pred)


# In[117]:


#Calculating the confusion matrix
make_confusion_matrix(y_val,y_val_pred)  ## check the model's performance on the validation set


# ### Neural Network with Adam Optimizer and Dropout

# In[216]:


backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)


# In[217]:


#Initializing the neural network
model_2 = Sequential()
#Adding the input layer with 32 neurons and relu as activation function
model_2.add(Dense(32,activation='relu',input_dim = X_train.shape[1]))
# Add dropout with ratio of 0.2 or any suitable value
model_2.add(Dropout(0.2))
# Add a hidden layer 
model_2.add(Dense(64,activation='relu'))
# Add a hidden layer 
model_2.add(Dense(64,activation='relu'))
# Add dropout with ratio of 0.1 or any suitable value
model_2.add(Dropout(0.1))
# Add a hidden layer 
model_2.add(Dense(32,activation='relu'))
# Add the number of neurons required in the output layer
model_2.add(Dense(1, activation = 'sigmoid'))


# In[218]:


#Use Adam as the optimizer.
optimizer = tf.keras.optimizers.Adam()

# uncomment one of the following lines to define the metric to be used
# metric = 'accuracy'
metric = keras.metrics.Recall()
# metric = keras.metrics.Precision()
# metric = keras.metrics.F1Score()


# In[219]:


## Compile the model with binary cross entropy as loss function and recall as the metric.
model_2.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=[metric])


# In[220]:


# Summary of the model
model_2.summary()


# In[221]:


#Fitting the ANN with batch_size = 32 and 100 epochs
history_2 = model_2.fit(
    X_train,y_train,
    batch_size=32,  ##specify the batch size.
    epochs=150, ##specify the # of epochs.
    verbose=1,
    validation_data=(X_val,y_val)
)


# **Loss function**

# In[222]:


#Plotting Train Loss vs Validation Loss
plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# From the above plot, we can observe that the train and validation curves are having smooth lines. Reducing the number of neurons and adding dropouts to the model worked, and the problem of overfitting was solved.

# In[223]:


#Plotting Train recall vs Validation recall
plt.plot(history_2.history['recall'])
plt.plot(history_2.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[224]:


#Predicting the results using best as a threshold
y_train_pred = model_2.predict(X_train)
y_train_pred = (y_train_pred > 0.5)
y_train_pred


# In[225]:


#Predicting the results using 0.5 as the threshold.
y_val_pred = model_2.predict(X_val)
y_val_pred = (y_val_pred > 0.5)
y_val_pred


# In[226]:


model_name = "NN with Adam & Dropout"

train_metric_df.loc[model_name] = recall_score(y_train,y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val,y_val_pred)


# **Classification report**

# In[227]:


#classification report
cr=classification_report(y_train,y_train_pred)
print(cr)


# In[228]:


#classification report
cr = classification_report(y_val,y_val_pred) ## check the model's performance on the validation set
print(cr)


# **Confusion matrix**

# In[229]:


#Calculating the confusion matrix
make_confusion_matrix(y_train, y_train_pred)


# In[230]:


#Calculating the confusion matrix
make_confusion_matrix(y_val,y_val_pred)  ## check the model's performance on the validation set


# ### Neural Network with Balanced Data (by applying SMOTE) and SGD Optimizer

# **Let's try to apply SMOTE to balance this dataset and then again apply hyperparamter tuning accordingly.**

# In[134]:


sm  = SMOTE(random_state=42)
#Fit SMOTE on the training data.
X_train_smote, y_train_smote= sm.fit_resample(X_train,y_train)
print('After UpSampling, the shape of train_X: {}'.format(X_train_smote.shape))
print('After UpSampling, the shape of train_y: {} \n'.format(y_train_smote.shape))


# Let's build a model with the balanced dataset

# In[135]:


backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)


# In[136]:


#Initializing the model
model_3 = Sequential()
#Add a input layer
model_3.add(Dense(32,activation='relu',input_dim = X_train_smote.shape[1]))
#Add a hidden layer
model_3.add(Dense(16,activation='relu'))
#Add a hidden layer 
model_3.add(Dense(16,activation='relu'))
#Add the required number of neurons in the output layer with a sigmoid activation function.
model_3.add(Dense(1, activation = 'sigmoid'))


# In[137]:


#Use SGD as the optimizer.
optimizer = tf.keras.optimizers.SGD(0.001)

# uncomment one of the following lines to define the metric to be used
# metric = 'accuracy'
metric = keras.metrics.Recall()
# metric = keras.metrics.Precision()
# metric = keras.metrics.F1Score()


# In[141]:


#Compile the model with binary cross entropy as loss function and recall as the metric
model_3.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=[metric])


# In[142]:


model_3.summary()


# In[143]:


#Fitting the ANN
history_3 = model_3.fit(
    X_train_smote, y_train_smote,
    batch_size=32, ## specify the batch size to use
    epochs=150, ## specify the number of epochs
    verbose=1,
    validation_data = (X_val,y_val)
)


# **Loss function**

# In[144]:


#Plotting Train Loss vs Validation Loss
plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[145]:


#Plotting Train recall vs Validation recall
plt.plot(history_3.history['recall'])
plt.plot(history_3.history['val_recall'])
plt.title('model recall')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[146]:


y_train_pred = model_3.predict(X_train_smote)
#Predicting the results using 0.5 as the threshold
y_train_pred = (y_train_pred > 0.5)
y_train_pred


# In[147]:


y_val_pred = model_3.predict(X_val)
#Predicting the results using 0.5 as the threshold
y_val_pred = (y_val_pred > 0.5)
y_val_pred


# In[148]:


model_name = "NN with SMOTE & SGD"

train_metric_df.loc[model_name] = recall_score(y_train_smote,y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val,y_val_pred)


# **Classification report**

# In[149]:


cr=classification_report(y_train_smote,y_train_pred)
print(cr)


# In[150]:


cr=classification_report(y_val,y_val_pred) ## check the model's performance on the validation set
print(cr)


# **Confusion matrix**

# In[151]:


#Calculating the confusion matrix
make_confusion_matrix(y_train_smote, y_train_pred)


# In[152]:


#Calculating the confusion matrix

make_confusion_matrix(y_val,y_val_pred) ## check the model's performance on the validation set


# ### Neural Network with Balanced Data (by applying SMOTE) and Adam Optimizer

# Let's build a model with the balanced dataset

# In[153]:


backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)


# In[160]:


#Initializing the model
model_4 = Sequential()
#Add a input layer 
model_4.add(Dense(32,activation='relu',input_dim = X_train_smote.shape[1]))
#Add a hidden layer 
model_4.add(Dense(16,activation='relu'))
#Add a hidden layer 
model_4.add(Dense(16,activation='relu'))
#Add the required number of neurons in the output layer and a suitable activation function.
model_4.add(Dense(1, activation = 'sigmoid'))


# In[161]:


model_4.summary()


# In[162]:


#Use Adam as the optimizer.
optimizer = tf.keras.optimizers.Adam()

# uncomment one of the following lines to define the metric to be used
# metric = 'accuracy'
metric = keras.metrics.Recall()
# metric = keras.metrics.Precision()
# metric = keras.metrics.F1Score()


# In[163]:


# Compile the model with binary cross entropy as loss function and recall as the metric
model_4.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=[metric])


# In[164]:


model_4.summary()


# In[165]:


#Fitting the ANN

history_4 = model_4.fit(
    X_train_smote,y_train_smote,
    batch_size=32, ## specify the batch size to use
    epochs=150,  ## specify the number of epochs
    verbose=1,
    validation_data = (X_val,y_val)
)


# **Loss function**

# In[169]:


#Plotting Train Loss vs Validation Loss
plt.plot(history_4.history['loss'])
plt.plot(history_4.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[172]:


#Plotting Train recall vs Validation recall
plt.plot(history_4.history['recall_1'])
plt.plot(history_4.history['val_recall_1'])
plt.title('model recall')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[173]:


y_train_pred = model_4.predict(X_train_smote)
#Predicting the results using 0.5 as the threshold
y_train_pred = (y_train_pred > 0.5)
y_train_pred


# In[174]:


y_val_pred = model_4.predict(X_val)
#Predicting the results using 0.5 as the threshold
y_val_pred = (y_val_pred > 0.5)
y_val_pred


# In[175]:


model_name = "NN with SMOTE & Adam"

train_metric_df.loc[model_name] = recall_score(y_train_smote,y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val,y_val_pred)


# **Classification report**

# In[176]:


cr=classification_report(y_train_smote,y_train_pred)
print(cr)


# In[177]:


cr=classification_report(y_val,y_val_pred) ## check the model's performance on the validation set
print(cr)


# **Confusion matrix**

# In[178]:


#Calculating the confusion matrix
make_confusion_matrix(y_train_smote, y_train_pred)


# In[179]:


#Calculating the confusion matrix
make_confusion_matrix(y_val,y_val_pred)  ## check the model's performance on the validation set


# ### Neural Network with Balanced Data (by applying SMOTE), Adam Optimizer, and Dropout

# In[180]:


backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)


# In[195]:


#Initializing the model
model_5 = Sequential()
# Add required no. of neurons to the input layer with relu as activation function
model_5.add(Dense(32,activation='relu',input_dim = X_train_smote.shape[1]))
# Add dropout rate
model_5.add(Dropout(0.2))
# Add required no. of neurons to the hidden layer with an activation function.
model_5.add(Dense(16,activation='relu'))
# Add dropout rate.
model_5.add(Dropout(0.1))
# Adding hidden layer with 8 neurons with relu as activation function
model_5.add(Dense(8,activation='relu'))
# Add the required number of neurons in the output layer with a suitable activation function.
model_5.add(Dense(1, activation = 'sigmoid'))


# In[196]:


# Complete the code to use Adam as the optimizer.
optimizer = tf.keras.optimizers.Adam()

# uncomment one of the following lines to define the metric to be used
# metric = 'accuracy'
metric = keras.metrics.Recall()
# metric = keras.metrics.Precision()
# metric = keras.metrics.F1Score()


# In[197]:


# Compile the model with binary cross entropy as loss function and recall as the metric
model_5.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=[metric])


# In[198]:


model_5.summary()


# In[199]:


history_5 = model_5.fit(
    X_train_smote,y_train_smote,
    batch_size=32, ## specify the batch size to use
    epochs=150, ## specify the number of epochs
    verbose=1,
    validation_data = (X_val,y_val))


# **Loss function**

# In[200]:


#Plotting Train Loss vs Validation Loss
plt.plot(history_5.history['loss'])
plt.plot(history_5.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[202]:


#Plotting Train recall vs Validation recall
plt.plot(history_5.history['recall_1'])
plt.plot(history_5.history['val_recall_1'])
plt.title('model recall')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[203]:


y_train_pred = model_5.predict(X_train_smote)
#Predicting the results using 0.5 as the threshold
y_train_pred = (y_train_pred > 0.5)
y_train_pred


# In[204]:


y_val_pred = model_5.predict(X_val)
#Predicting the results using 0.5 as the threshold
y_val_pred = (y_val_pred > 0.5)
y_val_pred


# In[205]:


model_name = "NN with SMOTE,Adam & Dropout"

train_metric_df.loc[model_name] = recall_score(y_train_smote,y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val,y_val_pred)


# **Classification report**

# In[206]:


cr=classification_report(y_train_smote,y_train_pred)
print(cr)


# In[207]:


#classification report
cr=classification_report(y_val,y_val_pred)  ## check the model's performance on the validation set
print(cr)


# **Confusion matrix**

# In[208]:


#Calculating the confusion matrix
make_confusion_matrix(y_train_smote, y_train_pred)


# In[209]:


#Calculating the confusion matrix
make_confusion_matrix(y_val,y_val_pred)  ## check the model's performance on the validation set


# ## Model Performance Comparison and Final Model Selection

# In[231]:


print("Training performance comparison")
train_metric_df


# In[232]:


print("Validation set performance comparison")
valid_metric_df


# In[233]:


train_metric_df - valid_metric_df


# In[234]:


y_test_pred = model_5.predict(X_test)    ## specify the best model
y_test_pred = (y_test_pred > 0.5)
print(y_test_pred)


# In[235]:


#print classification report
cr=classification_report(y_test,y_test_pred)
print(cr)


# In[236]:


#Calculating the confusion matrix
make_confusion_matrix(y_test,y_test_pred)


# ## Actionable Insights and Business Recommendations

# * The model "NN with SMOTE,Adam & Dropout" out of all the models performed well.
# * Adam converges faster as compared to SGD, Smote helped model to perform good on both the classes by avoiding the data imbalance and dropout helped in avoiding overfitting by making the model perform better on both the training and validation data.
# * The comparison was made based on the recall value on the training and validation dataset.
# * For all the models the batch size and no. of epochs are kept constant. Also, the no. of nuerons are preferred to be same in all the models but having slight variations in case of dropout models. 
# * This project is done from the point of view of implementation of neural network in real world scenerio. Also, it is performed to understand the model building, training and optimizing techniques.
# * Different combinations of no.of neurons in each layer, batch size, epochs can be used to obtain a final conclusion regarding the best model.
# 

# <font size=6 color='blue'>Power Ahead</font>
# ___
