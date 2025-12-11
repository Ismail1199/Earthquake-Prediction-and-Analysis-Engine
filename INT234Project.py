import pandas as pd
df = pd.read_csv('earthquake.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())

#-----------------------------------------------------------------------------------------

import numpy as np
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['updated'] = pd.to_datetime(df['updated'], errors='coerce')

numeric_cols = ['latitude','longitude','depth','mag','nst','gap','dmin','rms','horizontalError'
                ,'depthError','magError','magNst']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.drop_duplicates(inplace=True)

df['nst'] = df['nst'].fillna(df['nst'].median())
df['gap'] = df['gap'].fillna(df['gap'].median())
df['dmin'] = df['dmin'].fillna(df['dmin'].median())
df['horizontalError'] = df['horizontalError'].fillna(df['horizontalError'].median())
df['magError'] = df['magError'].fillna(df['magError'].median())
df['magNst'] = df['magNst'].fillna(df['magNst'].median())


cat_cols = ['magType','net','id','place','type','status','locationSource','magSource']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


df = df[df['mag'] > 0]
df = df[df['depth'] >= 0]
df = df[df['latitude'].between(-90,90)]
df = df[df['longitude'].between(-180,180)]

df_cleaned = df.copy()
df_cleaned.info()

#--------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(df['mag'], kde=True, bins=30)
plt.title("Magnitude Distribution")
plt.xlabel("Magnitude")
plt.ylabel("Count")
plt.show()

#-----------------------------------------------------------------------------------------

plt.figure(figsize=(12,5))
df.resample('D', on='time').size().plot()
plt.title("Earthquake Frequency per Day")
plt.ylabel("Count")
plt.show()

#-----------------------------------------------------------------------------------

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='depth', y='mag', alpha=0.6)
plt.title("Magnitude vs Depth")
plt.xlabel("Depth (km)")
plt.ylabel("Magnitude")
plt.show()

#------------------------------------------------------------------------------------

numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
plt.show()

#-----------------------------------------------------------------------------------

print(df.dtypes)

#-------------------------------------------------------------------------------