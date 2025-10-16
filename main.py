# Imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Base de données.

df = pd.read_csv(r"C:\Users\Rayane\Desktop\PAI Project\Titanic-Dataset.csv")

# Analyse de la bdd 

print(df.shape)       # Nombre de lignes et colonnes
print(df.info())      # Types de données et valeurs manquantes
print(df.describe())  # Statistiques numériques

print(df.isnull().sum())   # Nombre de valeurs manquantes par colonne

df['Age'].fillna(df['Age'].median(), inplace=True)

df = df.drop(["PassengerId", "Ticket", "Cabin", "Embarked"], axis=1)

# Histogramme des personnes ayant survécu en fonction de l'âge 

df[df["Survived"] == 1]["Age"].hist(grid=True, bins=20, color='green')
plt.title("Distribution des âges des survivants")
plt.xlabel("Âge")
plt.ylabel("Nombre de passagers")
plt.show()

# Répartition du nombre de survivant selon le sex 

survivants = df[df["Survived"] == 1]["Sex"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(
    survivants,
    labels=survivants.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#66b3ff", "#ff9999"]
)
plt.title("Répartition des survivants par sexe")
plt.show()

