#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as py


# In[13]:


df = pd.read_csv("./sales_data_with_discounts.csv")
print (data)


# In[14]:


num = df.dtypes[df.dtypes != 'object']
num


# In[47]:


df.describe()


# In[33]:


mean = df.mean(numeric_only=True)
mean


# In[44]:


median = df.median(numeric_only=True)
median


# In[42]:


mode = df.mode()
mode


# In[48]:


df.std(numeric_only=True)


# In[49]:


df.info()


# In[65]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df["Volume"], bins=20,kde=True)
plt.show()

sns.histplot(df["Avg Price"], bins=20,kde=True)
plt.show()

sns.histplot(df["Total Sales Value"], bins=20,kde=True)
plt.show()

sns.histplot(df["Discount Rate (%)"], bins=20,kde=True)
plt.show()

sns.histplot(df["Discount Amount"], bins=20,kde=True)
plt.show()

sns.histplot(df["Net Sales Value"], bins=20,kde=True)
plt.show()


# In[68]:


skew_vol = df["Volume"].skew()
print("Skewness of Volume:", skew_vol)


# In[74]:


plt.figure(figsize=(4,4))

plt.boxplot(df["Volume"], tick_labels=['Count'], patch_artist=True)
plt.title('Volume')
plt.ylabel('Count')
plt.show()

plt.boxplot(df["Avg Price"], tick_labels=['Count'], patch_artist=True)
plt.title('Volume')
plt.ylabel('Count')
plt.show()

plt.boxplot(df["Total Sales Value"], tick_labels=['Count'], patch_artist=True)
plt.title('Volume')
plt.ylabel('Count')
plt.show()

plt.boxplot(df["Discount Rate (%)"], tick_labels=['Count'], patch_artist=True)
plt.title('Volume')
plt.ylabel('Count')
plt.show()

plt.boxplot(df["Discount Amount"], tick_labels=['Count'], patch_artist=True)
plt.title('Volume')
plt.ylabel('Count')
plt.show()

plt.boxplot(df["Net Sales Value"], tick_labels=['Count'], patch_artist=True)
plt.title('Volume')
plt.ylabel('Count')
plt.show()


# In[77]:


cat_cols = df.select_dtypes(include=['object', 'category']).columns
print("Categorical columns:", cat_cols)


# In[57]:


import pandas as pd
import matplotlib.pyplot as plt

categorical_range = df.select_dtypes(include=['object', 'category'])
print(categorical_range)

cat_count = df["Date"].value_counts()
day_count = df["Day"].value_counts()
sku_count = df["SKU"].value_counts()
city_count = df["City"].value_counts()
bu_count = df["BU"].value_counts()
brand_count = df["Brand"].value_counts()
model_count = df["Model"].value_counts()

plt.figure(figsize=(15,4))

plt.barh(cat_count.index, cat_count.values)
plt.title('Category')
plt.ylabel('Count')
plt.show()

plt.barh(day_count.index, day_count.values)
plt.title('Day')
plt.ylabel('Count')
plt.show()

plt.barh(sku_count.index, sku_count.values)
plt.title('SKU')
plt.ylabel('Count')
plt.show()

plt.barh(city_count.index, city_count.values)
plt.title('City')
plt.ylabel('Count')
plt.show()

plt.barh(bu_count.index, bu_count.values)
plt.title('BU')
plt.ylabel('Count')
plt.show()

plt.barh(brand_count.index, brand_count.values)
plt.title('Brand')
plt.ylabel('Count')
plt.show()

plt.barh(model_count.index, model_count.values)
plt.title('Model')
plt.ylabel('Count')
plt.show()


# In[103]:


import matplotlib.pyplot as plt
import seaborn as sns

numerical_columns = df.select_dtypes(include=['int64','float64']).columns
#before
for i in numerical_columns:
    #print(i)
    plt.figure(figsize=(8, 4))    
    sns.histplot(df[i], bins=20, kde=True)

    plt.title(f'Distribution of {i} (Before)')
    plt.show()

#after
for i in numerical_columns:

    m = df[i].mean()
    s = df[i].std()
    standardized_data = (df[i] - m) / s
    #print(standardized_data)
    
    plt.figure(figsize=(8, 4))    
    sns.histplot(standardized_data, bins=20, kde=True, color='purple')
        
    plt.title(f'Distribution of {i} (After)')
    plt.axvline(0, color='red', linestyle='--')
    plt.show()

summary = df[numerical_columns].describe().loc[['mean', 'std']]
print(summary)


# In[96]:


#	Apply one-hot encoding to the categorical columns, creating binary (0 or 1) columns for each category
categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)

dummy = pd.get_dummies(df, columns=categorical_columns)

dummy.head()

