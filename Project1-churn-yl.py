# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
########################################################

# 1.Big Picture

## 1-import package:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## 2-load data:
testdf = pd.read_csv('C:\\Users\DELL\Desktop\\test.csv')
traindf = pd.read_csv('C:\\Users\DELL\Desktop\\train.csv')
data = testdf.append(traindf, ignore_index = True)

## 3-data preview:
print(data.head(5))
print(data.shape)
print(data.info())
data['Churn'] = data['Churn'].astype('int')
print(data.nunique())
print(data.isnull().sum())
print(data.describe())

## 4-visualization:
_,axss = plt.subplots(2,3, figsize=[20,10])
sns.boxplot(x='Churn', y ='Total day minutes', data=data, ax=axss[0][0])
sns.boxplot(x='Churn', y ='Total day calls', data=data, ax=axss[0][1])
sns.boxplot(x='Churn', y ='Total day charge', data=data, ax=axss[0][2])
sns.boxplot(x='Churn', y ='Total eve calls', data=data, ax=axss[1][0])
sns.boxplot(x='Churn', y ='Total eve charge', data=data, ax=axss[1][1])
sns.boxplot(x='Churn', y ='Total night calls', data=data, ax=axss[1][2])

corr_score = data[['Total day calls','Total day charge','Total day minutes']].corr()
print(corr_score)
sns.heatmap(corr_score)

#######################################################

# 2.feature preprocessing
## 1-encoding
data['International plan'] = (data['International plan'] == 'No').astype('int')
data['Voice mail plan'] = (data['Voice mail plan'] == 'No').astype('int')

## 2-drop
y = data['Churn']
drop_columns = ['Account length','Churn']
x = data.drop(drop_columns, axis = 1)

## 3-split data
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.25, stratify = y, random_state = 990128)

## 4-sclaing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

########################################################

# 3.model training and selection
## 1-train model & model selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
###Logistic Regression
classifier_logistic = LogisticRegression()
classifier_logistic.fit(X_train, y_train)
classifier_logistic.predict(X_test) #make prediction
classifier_logistic.score(X_test, y_test) #accuracy of test data
###K Nearest Neighbors
classifier_KNN = KNeighborsClassifier()
classifier_KNN.fit(X_train, y_train)
classifier_KNN.predict(X_test)
classifier_KNN.score(X_test, y_test)
###Random Forest
classifier_RF = RandomForestClassifier()
classifier_RF.fit(X_train, y_train)
classifier_RF.predict(X_test)
classifier_RF.score(X_test, y_test)

## 2-cross validation
model_names = ['Logistic Regression','KNN','Random Forest']
model_list = [classifier_logistic, classifier_KNN, classifier_RF]
count = 0
for classifier in model_list:
    cv_score = model_selection.cross_val_score(classifier, X_train, y_train, cv=5)
    print(cv_score)
    print('Model accuracy of ' + model_names[count] + ' is ' + str(cv_score.mean()))
    count += 1
    
## 3-grid search for Random Forest
model_selection.cross_val_score(classifier, X_train, y_train, cv=5)
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators' : [40,60,80]}
Grid_RF = GridSearchCV(RandomForestClassifier(),parameters, cv=5)
Grid_RF.fit(X_train, y_train)
best_RF_model = Grid_RF.best_estimator_
print(best_RF_model)

########################################################

# 4.model evaluation
## 1-confusion martix
from sklearn.metrics import confusion_matrix
def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print (classifier)
    print ("Accuracy is: " + str(accuracy))
    print ("precision is: " + str(precision))
    print ("recall is: " + str(recall))
def draw_confusion_matrices(confusion_matricies):
    class_names = ['Not','Churn']
    for cm in confusion_matrices: 
        classifier, cm = cm[0], cm[1] 
        cal_evaluation(classifier, cm)
        fig = plt.figure() 
        ax = fig.add_subplot(111) 
        cax = ax.matshow(cm, interpolation='nearest',cmap=plt.get_cmap('Reds')) 
        plt.title('Confusion matrix for ' + classifier) 
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
confusion_matrices = [("Random Forest", confusion_matrix(y_test,best_RF_model.predict(X_test)))]
draw_confusion_matrices(confusion_matrices)

## 2-AUC ROC
from sklearn.metrics import roc_curve
from sklearn import metrics
y_pred_rf = best_RF_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
###draw roc
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--') #draw diagonal
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - RF model')
plt.legend(loc='best')
plt.show()
###calculate auc
print(metrics.auc(fpr_rf,tpr_rf))

# To Xiangyang "Summer for thee, grant I may be"
########################################################













