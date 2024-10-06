#imports
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
random.seed(1)

#read the data set
year='2015'
brfss_2015_dataset= pd.read_csv("C:\Z Notes\A PROJECT\diabetes_012_health_indicators_BRFSS2015.csv")


#How many rows and columns
print(brfss_2015_dataset.shape)

#check that the data loaded in is in the correct format
pd.set_option('display.max_columns', 500)

#Let's see what the data looks like
print(brfss_2015_dataset.head())

#Check how many respondents have no diabetes, prediabetes or diabetes. Note the class imbalance!
print(brfss_2015_dataset.groupby(['Diabetes_012']).size())

# Change the diabetics 2 to a 1 and pre-diabetics 1 to a 0, so that we have 0 meaning non-diabetic and pre-diabetic and 1 meaning diabetic.
# We did this because from a clinical perspective, interventions or preventive measures aimed at reducing the risk of diabetes often target both
# pre-diabetic and non-diabetic individuals. Combining non-diabetic and pre-diabetic groups can increase the sample size and statistical power of
# the study, which is particularly beneficial if the number of pre-diabetic individuals is relatively small. This can improve the study's
# ability to detect significant differences or associations between variables. And for simplicity.

brfss_2015_dataset['Diabetes_012'] = brfss_2015_dataset['Diabetes_012'].replace({1:0})
brfss_2015_dataset['Diabetes_012'] = brfss_2015_dataset['Diabetes_012'].replace({2:1})

#Change the column name to Diabetes_binary
brfss_2015_dataset = brfss_2015_dataset.rename(columns = {'Diabetes_012': 'Diabetes_binary'})
print(brfss_2015_dataset.Diabetes_binary.unique())

#Show the change
print(brfss_2015_dataset.head())

#show class sizes
print(brfss_2015_dataset.groupby(['Diabetes_binary']).size())

#Separate the 0(No Diabetes) and 1&2(Pre-diabetes and Diabetes)
#Get the 1s
is1 = brfss_2015_dataset['Diabetes_binary'] == 1
brfss_5050_1 = brfss_2015_dataset[is1]
print(brfss_5050_1)

#Get the 0s
is0 = brfss_2015_dataset['Diabetes_binary'] == 0
brfss_5050_0 = brfss_2015_dataset[is0]
print(brfss_5050_0)

#Select the 39977 random cases from the 0 (non-diabetes group). we already have 35346 cases from the diabetes risk group
brfss_5050_0_rand1 = brfss_5050_0.take(np.random.permutation(len(brfss_5050_0))[:35346])
print(brfss_5050_0_rand1)

#Append the 39977 1s to the 39977 randomly selected 0s
brfss_5050 = brfss_5050_0_rand1._append(brfss_5050_1, ignore_index = True)
print(brfss_5050)

#Check that it worked. Now we have a dataset of 79,954 rows that is equally balanced with 50% 1 and 50% 0 for the target variable Diabetes_binary
brfss_5050.head()
brfss_5050.tail()

#See the classes are perfectly balanced now
print(brfss_5050.groupby(['Diabetes_binary']).size())

#Save binary dataset and 50-50 binary balanced dataset to csv
print(f'brfss_5050={brfss_5050.shape}',f'brfss_2015_dataset={brfss_2015_dataset.shape}')

#Save the 50-50 balanced dataset to csv
brfss_5050.to_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv', sep=",", index=False)

#Read the new datas set
data=pd.read_csv("C:\Z Notes\A PROJECT\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
df = data.copy()
print(df.shape)
df.head()

#There are no null values observed in the dataset. However, they are all loaded as "float" type. I will convert all columns except for "BMI" to "int"
# type as these values are binary / ordinal & not expected to take decimal values.
print(df.info())
print(" ")
print(df.nunique())

# Set all columns except for "BMI" to int type
columns_to_convert = df.columns.difference(['BMI'])
df[columns_to_convert] = df[columns_to_convert].astype(int)

#Let's see what the data looks like
print(df.info())
df.head()

#Since the data is balanced we move on
print(df.groupby(['Diabetes_binary']).size())


# List of binary, ordinal & numerical features
target = ['Diabetes_binary']
features_binary = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                   'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
features_ordinal = ['GenHlth', 'Age', 'Education', 'Income']
features_numerical = ['BMI', 'MentHlth', 'PhysHlth']

# Set up subplots for plotting

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))
fig.subplots_adjust(left=0.06,bottom=0.063,right=0.98,top=0.933,wspace=0.206,hspace=0.87)

# Loop through binary features and create plots
for i, feature in enumerate(features_binary):
    row, col = i // 3, i % 3
    sns.countplot(x=feature, hue='Diabetes_binary', data=df, ax=axes[row, col], palette='pastel')
    axes[row, col].set_title(f'Distribution of {feature}')

plt.show()

# Set up subplots for plotting
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,7))
fig.subplots_adjust(hspace=0.5)

# Loop through binary features and create plots
for i, feature in enumerate(features_ordinal):
    row, col = i // 2, i % 2
    sns.countplot(x=feature, hue='Diabetes_binary', data=df, ax=axes[row, col], palette='muted')
    axes[row, col].set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()

# Plot histplot and boxplot for each numerical feature
plt.figure(figsize=(15, 8))

for i, feature in enumerate(features_numerical, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df, x=feature, kde=True, bins=20, hue='Diabetes_binary', multiple='stack', palette='deep')
    plt.title(f'Distribution of {feature}')

    plt.subplot(2, 3, i + 3)
    sns.boxplot(x=df[feature], color='plum')
    plt.title(f'Boxplot of {feature}')

plt.tight_layout()
plt.show()

# Plot the correlation matrix heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()




















































