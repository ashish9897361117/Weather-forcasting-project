import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#
df = pd.read_csv(r"Cleaned_weather.csv")
df.head()
#
plt.figure(figsize=(8,5))
sns.histplot(df['MaxTemp'], bins=30, kde=True)
plt.title("Max Temperature Distribution")
plt.show()
#Rain Count
sns.countplot(x='RainTomorrow', data=df)
plt.title("Rain Tomorrow Distribution")
plt.show()
#Humidity vs Temperature
plt.figure(figsize=(8,5))
sns.scatterplot(x='Humidity3pm', y='MaxTemp', data=df)
plt.title("Humidity vs Temperature")
plt.show()
#Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
#Boxplot (Outliers check)
plt.figure(figsize=(8,5))
sns.boxplot(x=df['MaxTemp'])
plt.title("MaxTemp Outliers")
plt.show()
#Save the Dataset
df.to_csv("processed_weather.csv", index=False)
