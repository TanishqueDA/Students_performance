# Students_performance
Analysis the performance of students in Exam and graphical representation
# for some basic operations
import numpy as np
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
performance_data = pd.read_csv('Students_Performance.csv')
print(performance_data)

#For print first 5 surveys
performance_data.head()

#For print last 5 surveys
performance_data.tail()

#For sorting the data by parental level of education
performance_data.sort_values("parental level of education")

#Data Exploration
size = performance_data.size
shape = performance_data.shape
df_ndim = performance_data.ndim

print(size)
print(shape)
print(df_ndim)

print(performance_data.columns)

#Data Cleaning using Statistical Approch
performance_data.describe()

#check the no. of unique items present in the categorical column
performance_data.select_dtypes('object').nunique()

#check the percentage of missing data in each columns present in the data
no_of_columns = performance_data.shape[0]
percentage_of_missing_data = performance_data.isnull().sum()/no_of_columns
print(percentage_of_missing_data)

#bar plot
performance_data.plot.bar()

#box plot
performance_data.plot.box()

#comparison of all other attributes
# scatter plot between income and age
plt.scatter(performance_data['math score'], performance_data['reading score'])
plt.show()
  
# scatter plot between income and sales
plt.scatter(performance_data['math score'], performance_data['writing score'])
plt.show()
  
# scatter plot between sales and age
plt.scatter(performance_data['reading score'], performance_data['writing score'])
plt.show()

#check the Effect of Lunch on Student’s Performnce
performance_data[['lunch','gender','math score','writing score','reading score']].groupby(['lunch','gender']).agg('median')

#check the Effect of Test Preparation Course on Scores
performance_data[['test preparation course',
                  'gender',
                  'math score',
                  'writing score',
                  'reading score']].groupby(['test preparation course','gender']).agg('median')

#Data Visualizations
#Visualizing the number of male and female in the data set
plt.rcParams['figure.figsize'] = (15, 5)
sns.countplot(performance_data['gender'], palette = 'bone')
plt.title('Comparison of Males and Females', fontweight = 20)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

#Visualizing the different groups in the data set
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('ggplot')

sns.countplot(performance_data['race/ethnicity'], palette = 'pink')
plt.title('Comparison of various groups', fontweight = 30, fontsize = 20)
plt.xlabel('Groups')
plt.ylabel('count')
plt.show()

#Visualizing the different parental education levels
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')

sns.countplot(performance_data['parental level of education'], palette = 'Blues')
plt.title('Comparison of Parental Education', fontweight = 30, fontsize = 20)
plt.xlabel('Degree')
plt.ylabel('count')
plt.show()

#Visualizing Maths score
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('tableau-colorblind10')

sns.countplot(performance_data['math score'], palette = 'BuPu')
plt.title('Comparison of math scores', fontweight = 30, fontsize = 20)
plt.xlabel('score')
plt.ylabel('count')
plt.xticks(rotation = 90)
plt.show()

#Computing the total score for each student
import warnings
warnings.filterwarnings('ignore')

performance_data['total_score'] = performance_data['math score'] + performance_data['reading score'] + performance_data['writing score']

sns.distplot(performance_data['total_score'], color = 'magenta')

plt.title('comparison of total score of all the students', fontweight = 30, fontsize = 20)
plt.xlabel('total score scored by the students')
plt.ylabel('count')
plt.show()

#Computing percentage for each of the students
# importing math library to use ceil
from math import * 
import warnings
warnings.filterwarnings('ignore')

performance_data['percentage'] = performance_data['total_score']/3

for i in range(0, 1000):
    performance_data['percentage'][i] = ceil(performance_data['percentage'][i])

plt.rcParams['figure.figsize'] = (15, 9)
sns.distplot(performance_data['percentage'], color = 'orange')

plt.title('Comparison of percentage scored by all the students', fontweight = 30, fontsize = 20)
plt.xlabel('Percentage scored')
plt.ylabel('Count')
plt.show()

#feature engineering on the data to visualize and solve the dataset more accurately

#setting a passing mark for the students to pass on the three subjects individually
passmarks = 40

# creating a new column pass_math, this column will tell us whether the students are pass or fail
performance_data['pass_math'] = np.where(performance_data['math score']< passmarks, 'Fail', 'Pass')
performance_data['pass_math'].value_counts(dropna = False).plot.bar(color = 'red', figsize = (5, 3))

plt.title('Comparison of students passed or failed in maths')
plt.xlabel('status')
plt.ylabel('count')
plt.show()

#creating a new column pass_math, this column will tell us whether the students are pass or fail
performance_data['pass_reading'] = np.where(performance_data['reading score']< passmarks, 'Fail', 'Pass')
performance_data['pass_reading'].value_counts(dropna = False).plot.bar(color = 'green', figsize = (5, 3))

plt.title('Comparison of students passed or failed in maths')
plt.xlabel('status')
plt.ylabel('count')
plt.show()
     
# creating a new column pass_math, this column will tell us whether the students are pass or fail
performance_data['pass_writing'] = np.where(performance_data['writing score']< passmarks, 'Fail', 'Pass')
performance_data['pass_writing'].value_counts(dropna = False).plot.bar(color = 'blue', figsize = (5, 3))

plt.title('Comparison of students passed or failed in maths')
plt.xlabel('status')
plt.ylabel('count')
plt.show()

# computing the total score for each student

performance_data['total_score'] = performance_data['math score'] + performance_data['reading score'] + performance_data['writing score']

performance_data['total_score'].value_counts(normalize = True)
performance_data['total_score'].value_counts(dropna = True).plot.bar(color = 'cyan', figsize = (40, 8))

plt.title('comparison of total score of all the students')
plt.xlabel('total score scored by the students')
plt.ylabel('count')
plt.show()

# computing percentage for each of the students
# importing math library to use ceil
from math import * 

performance_data['percentage'] = performance_data['total_score']/3

for i in range(0, 1000):
  performance_data['percentage'][i] = ceil(performance_data['percentage'][i])

performance_data['percentage'].value_counts(normalize = True)
performance_data['percentage'].value_counts(dropna = False).plot.bar(figsize = (16, 8), color = 'red')

plt.title('Comparison of percentage scored by all the students')
plt.xlabel('percentage score')
plt.ylabel('count')
plt.show()

performance_data['status'] = performance_data.apply(lambda x : 'Fail' if x['pass_math'] == 'Fail' or 
                           x['pass_reading'] == 'Fail' or x['pass_writing'] == 'Fail'
                           else 'pass', axis = 1)

performance_data['status'].value_counts(dropna = False).plot.bar(color = 'gray', figsize = (3, 3))
plt.title('overall results')
plt.xlabel('status')
plt.ylabel('count')
plt.show()

#checking which student is fail overall
performance_data['status'] = performance_data.apply(lambda x : 'Fail' if x['pass_math'] == 'Fail' or 
                           x['pass_reading'] == 'Fail' or x['pass_writing'] == 'Fail'
                           else 'pass', axis = 1)

performance_data['status'].value_counts(dropna = False).plot.bar(color = 'gray', figsize = (3, 3))
plt.title('overall results')
plt.xlabel('status')
plt.ylabel('count')
plt.show()

#Assigning grades to the grades according to the following criteria :
# 0  - 40 marks : grade E
# 41 - 60 marks : grade D
# 60 - 70 marks : grade C
# 70 - 80 marks : grade B
# 80 - 90 marks : grade A
# 90 - 100 marks : grade Odef getgrade(percentage, status):

def getgrade(percentage, status):
  if status == 'Fail':
    return 'E'
  if(percentage >= 90):
    return 'O'
  if(percentage >= 80):
    return 'A'
  if(percentage >= 70):
    return 'B'
  if(percentage >= 60):
    return 'C'
  if(percentage >= 40):
    return 'D'
  else :
    return 'E'

performance_data['grades'] = performance_data.apply(lambda x: getgrade(x['percentage'], x['status']), axis = 1 )

performance_data['grades'].value_counts()

from sklearn.preprocessing import LabelEncoder

# creating an encoder
le = LabelEncoder()

# label encoding for test preparation course
performance_data['test preparation course'] = le.fit_transform(performance_data['test preparation course'])

# label encoding for lunch
performance_data['lunch'] = le.fit_transform(performance_data['lunch'])

# label encoding for race/ethnicity
# we have to map values to each of the categories
performance_data['race/ethnicity'] = performance_data['race/ethnicity'].replace('group A', 1)
performance_data['race/ethnicity'] = performance_data['race/ethnicity'].replace('group B', 2)
performance_data['race/ethnicity'] = performance_data['race/ethnicity'].replace('group C', 3)
performance_data['race/ethnicity'] = performance_data['race/ethnicity'].replace('group D', 4)
performance_data['race/ethnicity'] = performance_data['race/ethnicity'].replace('group E', 5)

# label encoding for parental level of education
performance_data['parental level of education'] = le.fit_transform(performance_data['parental level of education'])

#label encoding for gender
performance_data['gender'] = le.fit_transform(performance_data['gender'])

# label encoding for pass_math
performance_data['pass_math'] = le.fit_transform(performance_data['pass_math'])

# label encoding for pass_reading
performance_data['pass_reading'] = le.fit_transform(performance_data['pass_reading'])

# label encoding for pass_writing
performance_data['pass_writing'] = le.fit_transform(performance_data['pass_writing'])

# label encoding for status
performance_data['status'] = le.fit_transform(performance_data['status'])

#Data Preparation
#Splitting the dependent and independent variables
x = performance_data.iloc[:,:14]
y = performance_data.iloc[:,14]

print(x.shape)
print(y.shape)

#Splitting the data set into training and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 45)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# importing the MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# creating a scaler
mm = MinMaxScaler()

# feeding the independent variable into the scaler
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)

#Applying principal components analysis
from sklearn.decomposition import PCA

# creating a principal component analysis model
pca = PCA(n_components = None)

# feeding the independent variables to the PCA model
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# visualising the principal components that will explain the highest share of variance
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# creating a principal component analysis model
pca = PCA(n_components = 2)

# feeding the independent variables to the PCA model
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#Modelling
from sklearn.linear_model import  LogisticRegression

# creating a model
model = LogisticRegression()

# feeding the training data to the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# calculating the classification accuracies
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

from sklearn.tree import DecisionTreeClassifier

# creating a model
model = DecisionTreeClassifier()

# feeding the training data to the model
model.fit(x_train, y_train)

# predicting the x-test results
y_pred = model.predict(x_test)

# calculating the accuracies
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

#Printing the confusion matrix
from sklearn.metrics import confusion_matrix

# creating a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# printing the confusion matrix
plt.rcParams['figure.figsize'] = (8, 8)
sns.heatmap(cm, annot = True, cmap = 'Greens')
plt.title('Confusion Matrix for Logistic Regression', fontweight = 30, fontsize = 20)
plt.show()

#Random Forest
from sklearn.ensemble import RandomForestClassifier

# creating a model
model = RandomForestClassifier()

# feeding the training data to the model
model.fit(x_train, y_train)

# predicting the x-test results
y_pred = model.predict(x_test)

# calculating the accuracies
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

from sklearn.metrics import confusion_matrix

# creating a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# printing the confusion matrix
plt.rcParams['figure.figsize'] = (8, 8)
sns.heatmap(cm, annot = True, cmap = 'Reds')
plt.title('Confusion Matrix for Random Forest', fontweight = 30, fontsize = 20)
plt.show()

from pandas.plotting import radviz
fig, ax = plt.subplots(figsize=(12, 12))
new_df = x.copy()
new_df["status"] = y
radviz(new_df, "status", ax=ax, colormap="rocket")
plt.title('Radial Visualization for Target', fontsize = 10)
plt.show()

[Students_Performance.csv](https://github.com/TanishqueDA/Students_performance/files/11071427/Students_Performance.csv)
