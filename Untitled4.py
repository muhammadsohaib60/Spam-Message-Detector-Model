#!/usr/bin/env python
# coding: utf-8

# In[31]:


#Spam Message Detector Model


# In[12]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


data = pd.read_csv('emails_V2.csv')
data


# In[14]:


data.isnull().sum()


# In[15]:


#Removing rows with NULL values in text column


# In[16]:


data = data.dropna(subset = ['text'])
data.isnull().sum()


# In[17]:


# Hence data's NULL values are now cleaned, moreover I thought there would be a scenario where some of the rows would have only the spam column as NULL, and I would consider them as a test data set.


# In[18]:


data


# In[19]:


data.iloc[1024]


# In[20]:


data['text'][0]


# In[21]:


data['text'] = data['text'].str.lstrip('Subject:')


# In[22]:


data


# In[23]:


data['spam'].dtype


# In[24]:


#Converting spam column to int datatype


# In[25]:


data['spam'] = data['spam'].astype(int)
data['spam'].dtype


# In[26]:


data


# In[27]:


#Let's visualise the data


# In[28]:


sns.countplot(x = 'spam', data = data)


# In[29]:


#Converting textual data to numeric data


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer
import re


# In[31]:


for i in range(len(data['text'])):
    data['text'][i] = re.sub(r"[^a-zA-Z0-9]", ' ', data['text'][i])


# In[32]:


data['text'] = data['text'].str.lower()


# In[33]:


data


# In[34]:


corpus = data['text'].values


# In[35]:


vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(corpus)


# In[36]:


x


# In[37]:


y = data['spam']
y


# In[38]:


#Training the model


# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts


# In[40]:


x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.3, random_state = 42)


# In[41]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[42]:


model.score(x_train, y_train)


# In[43]:


#Confusion matrix


# In[44]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# In[45]:


y_pred = model.predict(x_test)


# In[46]:


cm = confusion_matrix(y_test, y_pred, labels = model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
disp.plot()
plt.show()


# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm


# In[48]:


data = pd.read_csv('emails_V2.csv')
data


# In[49]:


data.info()


# In[50]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.3, random_state = 42)


# In[ ]:


#Applying SVM algorithms


# In[55]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)


# In[56]:


print(classifier.score(x_test,y_test))


# In[57]:


# To read the csv files in arrays and dataframes.
import numpy as np 
import pandas as pd 


# In[62]:



data = pd.read_csv("emails_V2.csv", encoding = "latin-1")
# # encoding='latin-1' is used to download all special characters and everything in python. If there is no encoding on the data, it gives an error. Let's check the first five values.
data.head()


# In[63]:


data.isnull().sum()


# In[66]:



data.rename(columns= { 'v1' : 'class' , 'v2' : 'message'}, inplace= True)
data.head()


# In[ ]:


#Navie Bayes Algorthms


# In[68]:


import matplotlib.pyplot as plt
count =pd.value_counts(data["spam"], sort= True)
count.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.legend(loc='best')
plt.show()


# In[70]:


count.plot(kind = 'pie',autopct='%1.2f%%') # 1.2 is the decimal points for 2 places
plt.title('Pie chart')
plt.show()


# In[71]:


data.groupby('spam').describe()


# In[ ]:




