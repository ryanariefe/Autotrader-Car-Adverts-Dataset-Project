#!/usr/bin/env python
# coding: utf-8

# # PoDS Car Advert Project

# In[36]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(
    { "figure.figsize": (8, 6) },
    style='ticks',
    color_codes=True,
    font_scale=0.8
)
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = set(('retina', 'svg'))")

import warnings
warnings.filterwarnings('ignore')


# ### Importing and familiriasing with the Dataset

# In[3]:


# load the dataset adverts.csv
ads = pd.read_csv('adverts.csv')


# In[4]:


# Return the top 10 columns to familiarise with the dataframe(or 5 without argument)
ads.head(20)


# In[5]:


# Finding the zero mileage and null values and their error with new/used classification
zero_mileage = ads[(ads['mileage'] == 0)]
zero_mileage_used = ads[(ads['mileage'] == 0) & (ads['vehicle_condition'] == 'USED')]
null_mileage = ads['mileage'].isna()
len(zero_mileage), len(zero_mileage_used), null_mileage.sum()


# In[6]:


# Show unique values from fuel_type and count their occurences
ads['fuel_type'].unique


# In[7]:


# Selecting 20 random rows from the dataset
bi_fuel = ads[ads['fuel_type'] == 'Bi Fuel']
bi_fuel.sample(20)


# In[8]:


# Return the full overview of the dataframe including column names, counts, non-nulls, and data types
ads.info()


# In[9]:


# Use shape attribute to see the shape of the dataframe(wide/long)
ads.shape


# In[10]:


# Return the statistical summary of the numerical attributes
ads.describe()


# In[11]:


# Return the last 10 rows from the dataset
ads.tail(10)


# In[12]:


# Subsetting the dataset to rows with reg_code and empty year_of_registrations
# null_reg = ads.loc[(ads['reg_code'].notnull()) & (ads['year_of_registration']).isnull]
# Return 10 random samples from the subset dataframe above
# null_reg.sample(10)


# In[ ]:





# ## Data Pre-Processing

# In[13]:


# Looking at the null value
ads.isnull().sum()


# ### Missing Milage

# In[14]:


# Subsetting rows with missing milage values
no_mileage = ads[ads['mileage'].isnull()]
no_mileage.sample(10)


# In[15]:


# Filling missing mileage values with average from the average mileage of cars with the same year of registration
ads['mileage'] = ads['mileage'].fillna(
    ads.groupby('year_of_registration')['mileage'].transform('mean')
)
no_mileage.sample(10)


# In[16]:


# Using .dropna() to delete 19 rows with null mileage

ads = ads.dropna(subset = ['mileage'])


# In[17]:


ads.isnull().sum()


# ### Missing Year Of Registration

# In[18]:


# Subsetting rows where year_of_registration is null and reg_code is not null
registration_and_null_year = ads[ads['reg_code'].isnull() & ads['year_of_registration'].notnull()]
registration_and_null_year.head()
# There are 287 rows with missing year but has registration code


# In[19]:


# First we need to identify the registration code unique values

unique_reg = ads['reg_code'].unique()
unique_reg


# In[20]:


# Fill year_of_reg based on year on reg_code

# Define the letter to year dictionary
letter_to_year = {
    'A': 1983, 'B': 1984, 'C': 1985, 'D': 1986, 'E': 1987, 'F': 1988,
    'G': 1989, 'H': 1990, 'J': 1991, 'K': 1992, 'L': 1993, 'M': 1994,
    'N': 1995, 'P': 1996, 'R': 1997, 'S': 1998, 'T': 1999, 'V': 1999,
    'W': 2000, 'X': 2000, 'Y': 2001
}

# Converting lowercase to uppercase on reg_code
ads['reg_upper'] = ads['reg_code'].astype(str).str.upper().str.strip()

# Converting letter to year
mapped_years = ads['reg_upper'].map(letter_to_year)

# Filling the missing values on year_of_registration
ads['year_of_registration'] =ads['year_of_registration'].fillna(mapped_years)

# Dropping the temporary column for this task
ads.drop(columns=['reg_upper'], inplace=True)


# In[21]:


ads.info()


# In[22]:


# Replacing the missing year with numeric reg_code

# Define the function to convert numeric code to year
def numeric_code_to_year(code):
    if pd.isna(code):
        return None
    try:
        #converting possible wrong data type
        val = int(str(code).strip())
    except ValueError:
        return None
    #ignoring non-sensical number (eg. 94, 95)
    if val > 70:
        return None
    if val >=50:
        return 2000 + (val-50)
    elif val >= 2:
        return 2000 + val
    return None

# APPLYING THE FUNCTIOn

num_years = ads['reg_code'].apply(numeric_code_to_year)

ads['year_of_registration'] = ads['year_of_registration'].fillna(num_years)

ads[['reg_code', 'year_of_registration']].head()


# In[23]:


ads.info()
ads.isnull().sum()


# In[24]:


# Replacing missing values of the year of registration by using the median from the group of other rows with the same milage, standard_model, and price

19


# In[25]:


((ads['reg_code'].isna()) & (ads['year_of_registration'].isna())).sum


# In[26]:


# For rows with both missing year_of_registration and reg_code
# We can use the median price and standard model bins to place a best estimation

# Adding new feature price band by their standard_model
ads['price_level'] = ads.groupby('standard_model')['price'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
)

years_price = ads.groupby(['standard_model', 'price_level'])['year_of_registration'].transform('median')

# filling the missing year values with the information set up above
ads['year_of_registration'] = ads['year_of_registration'].fillna(years_price)

# Drop the temporary column for this task
ads.drop(columns=['price_level'], inplace=True)


# In[27]:


ads.isnull().sum()


# In[28]:


# We can delete the remaining rows with missing year_of_registration
# Assign a subset of rows with missing year_of_registration for later use
no_year = ads[ads['year_of_registration'].isna()]

ads = ads.dropna(subset = 'year_of_registration')


# In[29]:


# Filling missing registration code from the available year of registration



# ### Missing Body Type

# In[30]:


# We can fill missing body_type from getting mode of body_type from rows with the same standard_model and standard_make

# Getting mode from the grouped rows
mode_by_modelmake = ads.groupby(['standard_model', 'standard_make'])['body_type'].transform(
    lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
)

# Applying the mode to the rows with missing body_type
ads['body_type'] = ads['body_type'].fillna(mode_by_modelmake)


# ### Missing Fuel Type

# In[31]:


# Using SimpleImputer to fill in rows with missing fuel_type

from sklearn.impute import SimpleImputer

imp_mode = SimpleImputer(strategy='most_frequent')
ads['fuel_type'] = imp_mode.fit_transform(ads[['fuel_type']]).ravel()


# In[32]:


ads.isnull().sum()


# ### Missing Registration

# The information shown on the registration code(reg_code) column is only part of vehicle registration that indicate the year that the vehicle is registered. This information is useful to fill missing valus for year_of_registration. We can drop this column because there is not much analysis we can do with this feature.

# In[33]:


ads = ads.drop('reg_code', axis=1)


# In[34]:


ads.isnull().sum()


# ## Dealing with Outliers

# ### Identifying and Dealing with Outliers in Mileage

# In[39]:


# Plotting mileage data to identify outliers

sns.boxplot(ads['mileage'])


# In[42]:


# Subsetting mileage above 500000

mileage_over500000 = ads[ads['mileage'] > 500000]
ads['mileage'].describe()


# In[60]:


under_100000 = ads[ads['mileage']<100000]
plt.figure(figsize=(10,5))
sns.boxplot(x=under_100000['mileage'])
plt.title('Box Plot of Mileage (Subset: < 100,000 miles)')
plt.xlabel('Mileage (miles)')
plt.show()


# In[72]:


# Using Z-score to find the outlier and elimiate

# Getting the mileage z-score for each rows
mileage_zscore = (ads['mileage'] - ads['mileage'].mean())/ads['mileage'].std(ddof=0)
# Applying the threshold
zscore_plusmin3 = (mileage_zscore > -3) & (mileage_zscore < 3)
# Applying to the dataset
ads_cleaned = ads[zscore_plusmin3]

ads = ads_cleaned

sns.boxplot(ads['mileage'])
ads.info()


# ### Dealing with Price Ouliers

# In[86]:


# Plotting the price to see any outliers

BIN_WIDTH = 5000

plt.figure(figsize=(10, 6))
# Use binwidth to define clear, non-overlapping bins
sns.histplot(data=ads, x='price', binwidth=BIN_WIDTH, kde=True)

plt.title(f'Price Distribution with Clear Bins (Bin Width = {BIN_WIDTH})')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()


# In[89]:


# Using Z-score to find the outlier and elimiate

# Getting the mileage z-score for each rows
price_zscore = (ads['price'] - ads['price'].mean())/ads['price'].std(ddof=0)
# Applying the threshold
zscore_plusmin3 = (price_zscore > -3) & (price_zscore < 3)
# Applying to the dataset
ads_cleaned = ads[zscore_plusmin3]

ads = ads_cleaned


# In[93]:


bin_widht = 5000

plt.figure(figsize=(10, 6))
# Use binwidth to define clear, non-overlapping bins
sns.histplot(data=ads, x='price', binwidth=bin_widht, kde=True)

plt.title(f'Price Distribution with Clear Bins (Bin Width = {bin_widht})')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()


# In[121]:


# Treating rows with zero price as missing values

zero_price = ads[ads['price'] < 180]
zero_price.head()


# # Exploratory Data Analysis

# ### Fuel Type

# In[75]:


# Checking what are the fuel types available in this dataset
ads['fuel_type'].unique()


# In[100]:


# Creating new feature called energy_type
conditions = [
    (ads['fuel_type'].isin(['Diesel', 'Petrol', 'Bi Fuel'])),
    (ads['fuel_type'].isin(['Petrol Plug-in Hybrid', 'Diesel Hybrid', 'Petrol Hybrid', 'Electric', 'Diesel Plug-in Hybrid']))
]

choices = ['Fossil', 'Hybrid/Electric']

ads['energy_type'] = np.select(conditions, choices, default='Other')
ads.sample(10)


# In[81]:


# Plotting energy type to see the comparison
sns.histplot(ads['energy_type'])


# In[99]:


sum(ads['energy_type'] == 'Other')


# In[101]:


# Plotting price by their energy_type using violin plots

plt.figure(figsize=(10,6))
sns.violinplot(x='energy_type', y='price', data=ads)

# Setting lable and the axis
plt.title('Price Distribution by Energy Type')
plt.xlabel('Energy Type')
plt.ylabel('Price')
plt.grid(axis='y', alpha=0.5)
plt.show()


# In[108]:


# Plotting mileage and price of Subaru Outback to see the relation between the two features
subaru_outback = ads[(ads['standard_make'] == 'Subaru') & (ads['standard_model'] == 'Outback')]

sns.scatterplot(
    x='mileage',
    y='price',
    data=subaru_outback
)

plt.title('Price Vs. Mileage for Subaru Outback')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.show()

