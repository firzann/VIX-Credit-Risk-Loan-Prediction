#!/usr/bin/env python
# coding: utf-8

# In[35]:





# In[13]:


import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
warnings.filterwarnings('ignore')

df = pd.read_csv('loan_data_2007_2014.csv')


# ### Check Duplicate

# In[14]:


df.duplicated()


# ### Check Null

# In[15]:


df.isna().sum()


# In[16]:


df.loan_status.value_counts()

status = []

for index , kolom in df.iterrows():
    if 'Current' in  kolom['loan_status']:
        status.append(1)
    elif 'Fully Paid'in  kolom['loan_status']:
        status.append(1)
    elif 'In Grace Period 'in  kolom['loan_status']:
        status.append(1)
    else:
        status.append(0)
        
df['Status'] = status

df.head()


# In[17]:


df_penting = df[['loan_amnt', 'int_rate', 'installment', 'grade', 'annual_inc', 'issue_d', 'pymnt_plan', 'delinq_2yrs', 'mths_since_last_delinq',
              'open_acc', 'revol_bal', 'revol_util', 'total_pymnt', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'collections_12_mths_ex_med',
              'acc_now_delinq', 'tot_cur_bal', 'total_rev_hi_lim', 'Status']]
df_penting.info()


# In[18]:


df_penting.isna().sum()


# In[19]:


df_penting['mths_since_last_delinq'].fillna(df_penting['mths_since_last_delinq'].median(), inplace = True)
df_penting['open_acc'].fillna(df_penting['open_acc'].median(), inplace = True)
df_penting['revol_util'].fillna(df_penting['revol_util'].median(), inplace = True)
df_penting['collections_12_mths_ex_med'].fillna(df_penting['collections_12_mths_ex_med'].median(), inplace = True)
df_penting['acc_now_delinq'].fillna(df_penting['acc_now_delinq'].median(), inplace = True)
df_penting['annual_inc'].fillna(df_penting['annual_inc'].median(), inplace = True)
df_penting['tot_cur_bal'].fillna(df_penting['tot_cur_bal'].median(), inplace = True)
df_penting['total_rev_hi_lim'].fillna(df_penting['total_rev_hi_lim'].median(), inplace = True)
df_penting['delinq_2yrs'].fillna(df_penting['delinq_2yrs'].median(), inplace = True)


# In[20]:


df_penting.isna().sum()


# ## Heatmap

# In[21]:


plt.figure(figsize = (15,13))
sns.heatmap(df_penting.corr(), annot = True, fmt = '.2f');


# ### Feature Selection (2)

# In[22]:


df_dropped = df_penting.drop(columns = ['installment', 'total_pymnt', 'revol_bal', 'collection_recovery_fee', 'issue_d'])
plt.figure(figsize = (15,13))
sns.heatmap(df_dropped.corr(), annot = True, fmt = '.2f');


# In[23]:


df_dropped.describe()


# ## Handling Outliers

# In[57]:


from scipy import stats

print('Jumlah baris sebelum memfilter outlier:', len(df_dropped))

filtered_entries = np.array([True] * len(df))

for col in ['loan_amnt', 'int_rate', 'annual_inc', 'delinq_2yrs',
       'mths_since_last_delinq', 'open_acc', 'revol_util', 'recoveries',
       'last_pymnt_amnt', 'collections_12_mths_ex_med', 'acc_now_delinq',
       'tot_cur_bal', 'total_rev_hi_lim', 'Status']:
    
    zscore = abs(stats.zscore(df_dropped[col])) #  absolute z-scorenya
    filtered_entries = (zscore < 3) & filtered_entries # keep yang kurang dari 3 absolute z-scorenya
    
df_dropped = df_dropped[filtered_entries] # ambil yang z-scorenya dibawah 3

print('Jumlah baris setelah memfilter outlier:', len(df_dropped))


# ## Scalling

# In[25]:


df_dropped.info()


# In[26]:


df_clean = df_dropped.drop(columns = ['collections_12_mths_ex_med', 'acc_now_delinq'])
plt.figure(figsize = (15,13))
sns.heatmap(df_clean.corr(), annot = True, fmt = '.2f')


# In[27]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
df_clean['loan_amnt_norm'] = MinMaxScaler().fit_transform(df_clean['loan_amnt'].values.reshape(len(df_clean), 1))
df_clean['int_rate_norm'] = MinMaxScaler().fit_transform(df_clean['int_rate'].values.reshape(len(df_clean), 1))
df_clean['annual_inc_std'] = StandardScaler().fit_transform(df_clean['annual_inc'].values.reshape(len(df_clean), 1))
df_clean['delinq_2yrs_norm'] = MinMaxScaler().fit_transform(df_clean['delinq_2yrs'].values.reshape(len(df_clean), 1))
df_clean['mths_since_last_delinq_norm'] = MinMaxScaler().fit_transform(df_clean['mths_since_last_delinq'].values.reshape(len(df_clean), 1))
df_clean['open_acc_norm'] = MinMaxScaler().fit_transform(df_clean['open_acc'].values.reshape(len(df_clean), 1))
df_clean['revol_util_norm'] = MinMaxScaler().fit_transform(df_clean['revol_util'].values.reshape(len(df_clean), 1))
df_clean['recoveries_std'] = StandardScaler().fit_transform(df_clean['recoveries'].values.reshape(len(df_clean), 1))
df_clean['last_pymnt_amnt_std'] = StandardScaler().fit_transform(df_clean['last_pymnt_amnt'].values.reshape(len(df_clean), 1))
df_clean['tot_cur_bal_std'] = StandardScaler().fit_transform(df_clean['tot_cur_bal'].values.reshape(len(df_clean), 1))
df_clean['total_rev_hi_lim_std'] = StandardScaler().fit_transform(df_clean['total_rev_hi_lim'].values.reshape(len(df_clean), 1))


# In[28]:


df_clean.info()


# In[29]:


#drop columns orginal before scalling
df_cleaning = df_clean.drop(columns = ['loan_amnt', 'int_rate', 'annual_inc', 'delinq_2yrs', 'mths_since_last_delinq', 'open_acc',
                                       'revol_util', 'recoveries', 'last_pymnt_amnt', 'tot_cur_bal', 'total_rev_hi_lim'])
df_cleaning.info()


# In[30]:


plt.figure(figsize = (15,13))
sns.heatmap(df_cleaning.corr(), annot = True, fmt = '.2f')


# ## Feature Encoding

# In[31]:


for cat in ['grade', 'pymnt_plan']:
    onehots = pd.get_dummies(df_cleaning[cat], prefix=cat)
    df_cleaning = df_cleaning.join(onehots)


# In[32]:


df_cleaning = df_cleaning.drop(columns = ['pymnt_plan', 'grade'])
df_cleaning.info()


# In[33]:


plt.figure(figsize = (15,13))
sns.heatmap(df_cleaning.corr(), annot = True, fmt = '.2f');


# ## Modelling

# In[34]:


X = df_cleaning.drop(labels=['Status'],axis=1)
y = df_cleaning[['Status']]


# In[35]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify=y,random_state = 42)


# In[36]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def eval_classification(model, xtrain, ytrain, xtest, ytest):
    ypred = model.predict(xtest)
    print("Accuracy (Test Set): %.2f" % accuracy_score(ytest, ypred))
    print("Precision (Test Set): %.2f" % precision_score(ytest, ypred))
    print("Recall (Test Set): %.2f" % recall_score(ytest, ypred))
    print("F1-Score (Test Set): %.2f" % f1_score(ytest, ypred))
    
    y_pred_proba = model.predict_proba(xtest)
    print("AUC: %.2f" % roc_auc_score(ytest, y_pred_proba[:, 1]))


# In[37]:


def show_feature_importance(model):
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 8))
    ax.invert_yaxis()

    plt.xlabel('score')
    plt.ylabel('feature')
    plt.title('feature importance score')


# In[38]:


def show_best_hyperparameter(model, hyperparameters):
    for key, value in hyperparameters.items() :
        print('Best '+key+':', model.get_params()[key])


# ## Logistic Regression

# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# List Hyperparameters yang akan diuji
penalty = ['l2','l1','elasticnet','none']
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000] # Inverse of regularization strength; smaller values specify stronger regularization.
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga','none']
class_weight = [{0: 1, 1: 1},
                {0: 1, 1: 2}, 
                {0: 1, 1: 3},
                {0: 1, 1: 4},
                'none']
hyperparameters = dict(penalty=penalty, C=C,class_weight=class_weight,solver=solver)

# Inisiasi model
logres = LogisticRegression(random_state=42) # Init Logres dengan Gridsearch, cross validation = 5
model = RandomizedSearchCV(logres, hyperparameters, cv=5, random_state=42, scoring='recall')

# Fitting Model & Evaluation
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
eval_classification(model, X_train, y_train, X_test, y_test)


# ## Evaluation

# In[42]:


y_pred_train = model.predict(X_train)
y_pred_train


# In[43]:


from sklearn.metrics import roc_auc_score #ini gapake predict_proba
roc_auc_score(y_test, y_pred)


# In[44]:


eval_classification(model, X_train, y_train, X_test, y_test)


# In[45]:


print("Recall (Train Set): %.2f" % recall_score(y_train, y_pred_train))
print('Train score: ' + str(model.score(X_train, y_train))) #tes accuracy
print('Test score:' + str(model.score(X_test, y_test))) #test accuracy


# ## Decision Tree

# In[46]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)
eval_classification(dt, X_train, y_train, X_test, y_test)


# In[47]:


y_pred_train = dt.predict(X_train)
print("Recall (Train Set): %.2f" % recall_score(y_train, y_pred_train))


# In[49]:


print('Train score: ' + str(dt.score(X_train, y_train))) #accuracy
print('Test score:' + str(dt.score(X_test, y_test))) #accuracy


# ## Random Forest

# In[50]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
eval_classification(rf, X_train, y_train, X_test, y_test)


# In[51]:


y_pred_train = rf.predict(X_train)
print("Recall (Train Set): %.2f" % recall_score(y_train, y_pred_train))


# In[52]:


print('Train score: ' + str(rf.score(X_train, y_train))) #accuracy
print('Test score:' + str(rf.score(X_test, y_test))) #accuracy


# ## KNN

# In[53]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = knn.predict(X_test)
eval_classification(knn, X_train, y_train, X_test, y_test)


# In[54]:


ins = df.groupby(['loan_status']).agg({'id' : 'count'}).sort_values(['id'], ascending = False).reset_index()
ins.columns = ['loan_status', 'frequency']
ins['percentage %'] = round(ins['frequency']*100/sum(ins['frequency']),2)
ins

