#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[4]:


df=pd.read_csv("C:\\Users\\91762\\Downloads\\exams.csv")


# In[5]:


print(df.head())


# In[6]:


print(df.isnull().sum())


# In[7]:


print(df.info())


# In[8]:


print(df.describe())


# In[9]:


df = df.dropna()
df


# In[10]:


X = df.drop(['math score', 'reading score', 'writing score'], axis=1)
y = df[['math score']]


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[13]:


df


# In[14]:


correlation_matrix = df.corr()
correlation_matrix


# In[15]:


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[18]:


df = pd.read_csv("C:\\Users\\91762\\Downloads\\exams.csv")


# In[19]:


df = df.dropna()
df


# In[20]:


X = df.drop(['math score', 'reading score', 'writing score'], axis=1)
y = df[['math score']]


# In[21]:


label_encoder = LabelEncoder()
X["gender"] = label_encoder.fit_transform(X["gender"])
X["race/ethnicity"] = label_encoder.fit_transform(X["race/ethnicity"])
X["parental level of education"] = label_encoder.fit_transform(X["parental level of education"])
X["lunch"] = label_encoder.fit_transform(X["lunch"])
X["test preparation course"] = label_encoder.fit_transform(X["test preparation course"])


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()


# In[24]:


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[25]:


classifier = LogisticRegression()


# In[26]:


classifier.fit(X_train_scaled, y_train)


# In[27]:


y_pred = classifier.predict(X_test)


# In[28]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


# In[29]:


print("Logistic Regression:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[31]:


df = pd.read_csv("C:\\Users\\91762\\Downloads\\exams.csv")


# In[32]:


df = df.dropna()
df


# In[33]:


X = df.drop(['math score', 'reading score', 'writing score'], axis=1)
y = df[['math score']]


# In[34]:


label_encoder = LabelEncoder()
X["gender"] = label_encoder.fit_transform(X["gender"])
X["race/ethnicity"] = label_encoder.fit_transform(X["race/ethnicity"])
X["parental level of education"] = label_encoder.fit_transform(X["parental level of education"])
X["lunch"] = label_encoder.fit_transform(X["lunch"])
X["test preparation course"] = label_encoder.fit_transform(X["test preparation course"])


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[36]:


y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()


# In[37]:


classifier = DecisionTreeClassifier()


# In[38]:


classifier.fit(X_train, y_train)


# In[39]:


y_pred = classifier.predict(X_test)


# In[40]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


# In[41]:


print("Decision Tree:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[42]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[43]:


df = pd.read_csv("C:\\Users\\91762\\Downloads\\exams.csv")


# In[44]:


df = df.dropna()
df


# In[45]:


X = df.drop(['math score', 'reading score', 'writing score'], axis=1)
y = df[['math score']]


# In[46]:


label_encoder = LabelEncoder()
X["gender"] = label_encoder.fit_transform(X["gender"])
X["race/ethnicity"] = label_encoder.fit_transform(X["race/ethnicity"])
X["parental level of education"] = label_encoder.fit_transform(X["parental level of education"])
X["lunch"] = label_encoder.fit_transform(X["lunch"])
X["test preparation course"] = label_encoder.fit_transform(X["test preparation course"])


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[48]:


y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()


# In[49]:


classifier = RandomForestClassifier()


# In[50]:


classifier.fit(X_train, y_train)


# In[51]:


y_pred = classifier.predict(X_test)


# In[52]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


# In[53]:


print("Random Forest:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[54]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[55]:


df = pd.read_csv("C:\\Users\\91762\\Downloads\\exams.csv")


# In[56]:


df = df.dropna()
df


# In[57]:


X = df.drop(['math score', 'reading score', 'writing score'], axis=1)
y = df[['math score']]


# In[58]:


label_encoder = LabelEncoder()
X["gender"] = label_encoder.fit_transform(X["gender"])
X["race/ethnicity"] = label_encoder.fit_transform(X["race/ethnicity"])
X["parental level of education"] = label_encoder.fit_transform(X["parental level of education"])
X["lunch"] = label_encoder.fit_transform(X["lunch"])
X["test preparation course"] = label_encoder.fit_transform(X["test preparation course"])


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[60]:


y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()


# In[61]:


classifier = SVC()


# In[62]:


classifier.fit(X_train, y_train)


# In[63]:


y_pred = classifier.predict(X_test)


# In[64]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


# In[65]:


print("Support Vector Machines (SVM):")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[66]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score


# In[67]:


df = pd.read_csv("C:\\Users\\91762\\Downloads\\exams.csv")


# In[68]:


df = df.dropna()
df


# In[69]:


X = df.drop(['math score', 'reading score', 'writing score'], axis=1)
y = df[['math score']]


# In[70]:


label_encoder = LabelEncoder()
X["gender"] = label_encoder.fit_transform(X["gender"])
X["race/ethnicity"] = label_encoder.fit_transform(X["race/ethnicity"])
X["parental level of education"] = label_encoder.fit_transform(X["parental level of education"])
X["lunch"] = label_encoder.fit_transform(X["lunch"])
X["test preparation course"] = label_encoder.fit_transform(X["test preparation course"])


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[72]:


y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()


# In[73]:


classifier = KNeighborsClassifier()


# In[74]:


classifier.fit(X_train, y_train)


# In[75]:


y_pred = classifier.predict(X_test)


# In[76]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


# In[77]:


print("K-Nearest Neighbors (KNN):")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[85]:


plt.figure(figsize=(6, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:




