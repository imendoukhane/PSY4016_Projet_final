#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import pipeline
from sklearn import decomposition
from sklearn import naive_bayes
from scipy.stats import f_oneway


# <div class="alert alert-success">Ouverture de la base de données
# 

# In[11]:


# Classe permettant de lire les fichiers adéquatement selon s'ils sont sous format csv ou excel
class READFILE:
    
    def __init__(self, file):
        self.file = file
        if file.endswith("csv"):
            self.read_csv()
        elif file.endswith("xlsx"):
            self.read_excel()

    def read_csv(self):
        return pandas.read_csv(self.file)

    def read_excel(self):
        return pandas.read_excel(self.file)
    
    def verif_presence(self):
        return os.path.exists(self.file)


# In[12]:


# Ouverture de mon fichier excel (utilisation de classe)

my_data = "/Users/imendoukhane/Library/Mobile Documents/com~apple~Numbers/Documents/Human_life_Expectancy_lucas.xlsx"
p = READFILE(my_data)
df = p.read_excel()


# In[13]:


df.head()


# # <div class="alert alert-success"> Organisation/Nettoyage de la base de données
# <li> Conserver uniquement les données nationales 
# <li> Conserver uniquement les données de 2019
# <li> Créer une colonne "Statut" dans laquelle sera stockée le niveau de développement de chaque pays
# <li> Sélectionner un sous-ensemble de pays développés (n = 20) et de pays en voie de développement (n = 20)
#  

# In[14]:


# Objectif : Conserver uniquement les données des régions nationales

new_df = df.loc[df.Level == "National", :]


# In[15]:


# Extraire toutes les colonnes qui ont université de montréal comme établissement ET qui sont inférieur à 100 000

new = new_df.drop(['1990', '1991', '1992',
       '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
       '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
       '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'],axis=1)


# In[16]:


new


# In[17]:



# Créer une nouvelle colonne "Statut" pour catégoriser mes pays (développés vs en voie de développement)

new["Statut"] = ""


# In[18]:


new


# # <div class="alert alert-success"> Traitement des données (analyses statistiques)
# <li> Sélectionner les pays sur lesquels les analyses seront conduites
# <li> Vérifier s'il existe des données manquantes
# <li> Définir le statut (niveau de développement) des pays à l'étude
# <li> AA supervisé + graphique
# <li> AA non-supervisé + graphique
# <li> Visualiser la distribution de mes données
# <li> Vérifier le respect des postulats
# <li> ANOVA simple

# # <div class="alert alert-success">  Rationel des analyses
#     
# - Au lieu de tester la différence d'espérance de vie entre tous les pays développés (Europe, Amérique) et tous les pays en voie de développement (Afrique, Asie), je le ferais uniquement auprès d'un sous-ensemble de pays plus restreint. Sachant que je ne peux pas non classifier manuellement les pays selon leur statut de développement, j'ai sélectionné 40 pays, soit 20 pays en voie de développement (Afrique, Asie) et 20 pays développés (Europe, Amérique), sur lesquelles mes analyses seront conduites :
# 
#     Pays développés
#  - Canada
#  - Norway
#  - United States
#  - Japan
#  - Germany
#  - Finland
#  - France
#  - Italy
#  - Sweden
#  - Australia
#  - Liechtenstein
#  - Greece
#  - Netherlands
#  - Portugal
#  - Austria
#  - Ireland
#  - Slovenia
#  - United Kingdom
#  - Belgium
#  - Slovakia
#  
#      Pays en voie de développement
#  - Ethiopia
#  - Togo
#  - Mali
#  - Zambia
#  - Honduras
#  - Guatemala
#  - Venezuela
#  - Iraq
#  - Yemen
#  - Vietnam
#  - Niger
#  - Chad
#  - Comores
#  - Nepal
#  - Afghanistan
#  - Djibouti
#  - Angola
#  - Burkina Faso
#  - Somalia
#  - Syria
#  - Algeria
#  
#  
# Rappel de l'objectif : Comparer l'espérance de vie moyenne entre les pays développés et les pays en voie de développement
# 

# In[19]:


# Je vais  sélectionner 10 pays développés et 10 pays en voie de développement

database_a= new[new['Country'].str.contains("Canada|Norway|United States|Japan|Germany|Finland|France|Italy|Sweden|Australia|Ethiopia|Togo|Mali|Zambia|Honduras|Guatemala|Venezuela|Iraq|Yemen|Algeria") == True]
            


# In[20]:


database_a


# In[21]:


# Ajouter des pays

database_b = new[new['Country'].str.contains("Liechtenstein|Greece|Netherlands|Portugal|Austria|Ireland|Slovenia|United Kingdom|Belgium|Slovakia|Niger|Chad|Comores|Nepal|Afghanistan|Djibouti|Angola|Burkina Faso|Somalia|Syria")==True]


# In[22]:


database_b


# In[23]:


# Combiner l'ensemble des données qui m'intéressent en 1 seule cadre de données

frames = [database_a, database_b]
df = pd.concat(frames)


# In[24]:


df


# In[25]:


# Renommer la colonne d'espérance de vie en 2019

df.rename(columns = {'2019':'Espérance'}, inplace = True)


# In[26]:


# Réinitialiser les index 

df = df.reset_index(drop=True)


# In[27]:


df


# In[28]:


# Vérifier s'il existe des données manquantes dans ma base de données 

x = df.apply(lambda x: x.isnull().any(),axis=0)        # Fonction anonyme
print(x)

# Conclusion : Il existe au moins une donnée manquante dans la colonne d'espérance de vie


# In[29]:


# Allons voir dans quelle position cette donnée manquante se trouve, afin de supprimer la ligne au complet
x = df.apply(lambda x: x.isnull().any(),axis=1)
print(x)


# In[30]:


# Quel est le pays pour lequel nous n'avons pas de données d'espérance de vie ?
nom = df["Country"][38]   # NaN se trouve à la position 38, comme déterminé à la cellule précédente
print(f'Le pays pour lequel nous n\'avons pas de données d\'espérance de vie est {nom}')             #style f-Strings : On peut ajouter directement les variables entre les accolades dans la position souhaitée


# In[31]:


# Supprimer la Syrie de la base de données

df.drop([38], axis=0, inplace=True)
df = df.reset_index(drop=True)


# In[32]:


df


# In[34]:


# Définir le statut des pays à l'étude 
df.at[[1,2,4,5,6,10,11,13,14,16,22,23,27,28,29,31,34,35,36,38], "Statut"] = "Dévelopé"       # 20 pays développés

df.at[[0,3,7,8,9,12,15,17,18,19,20,21,24,25,26,30,32,33,37,], "Statut"] = "Non-dévelopé"     #19 pays en voie de développement


# In[35]:


df


# In[36]:


groupby_statut = df.groupby(["Statut"])


# In[37]:


# Trouver la moyenne d'Espérance de vie des pauys dévelopés et non-dévelopés
groupby_statut.mean()


# In[38]:


# Créer une copie de ma base de données pour pouvoir la modifier sans perdre l'original
df_copy = df.copy(deep=True)


# In[39]:


# Je veux prédire ma colonne "Statut" sur la base de la colonne "Espérance de vie"

# Je veux enlever toutes les autres colonnes, puisqu'elles ne me sont pas d'intérêt
  
df_copy.drop(['Country', 'Country_Code', 'Level', 'Region'], axis=1, inplace=True)


# In[40]:


df_copy


# In[41]:


# Extraire seulement les caractéristiques que j'ai de besoin

X_database = df_copy.drop('Statut', axis=1)
y_database = df_copy['Statut']


# In[42]:


X_database  # Matrices de caractéristiques


# In[43]:


y_database  # Ce que je cherche à prédire à partir de la matrice de caractéristiques


# In[62]:


# Algorithme de validation croisée

from sklearn.model_selection import train_test_split   # Pour créer mon set d'entraînement et de test

Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X_database, y_database,
                                                random_state=1)

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import LogisticRegression
X= Xtrain    # Données de df
Y= ytrain    # Données de df
logreg=LogisticRegression()
kf=KFold(n_splits=5)
score=cross_val_score(logreg,X,Y,cv=kf)
print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))


# In[67]:


# AA supervisé 

# 1. choisir la classe de modèle
from sklearn.naive_bayes import GaussianNB

# 2. choisir les hyperparameters du modèle
model = naive_bayes.GaussianNB()

# 3.  arrangées les données
from sklearn.model_selection import train_test_split   # Pour créer mon set d'entraînement et de test

Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X_database, y_database,
                                                random_state=1)
# 4. adapter le modèle aux données d'entraînement
model.fit(Xtrain, ytrain)

# 5. appliquer le modèle à de nouvelles données (données tests)
y_model = model.predict(Xtest)

# Utilisez l'utilitaire accuracy_score
# pour afficher la fraction d'étiquettes prédites correspondant à leur valeur réelle:

print(f'Prediction accuracy {sklearn.metrics.accuracy_score(ytest, y_model):.2%}')


# In[68]:


# Afficher les résultats de ma prédiction avec une matrice de confusion
import seaborn as sns
import matplotlib.pyplot as plt     

mat = confusion_matrix(ytest, y_model)
ax= plt.subplot()
sns.heatmap(mat, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

#labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Dévelopé', 'Non-dévelopés']); ax.yaxis.set_ticklabels(['Dévelopés', 'Non-dévelopés']);


# In[ ]:


# AA non supervisé (PCA) + Pipeline

from sklearn import pipeline
from sklearn import decomposition
from sklearn import naive_bayes

# unscaled_clf = sklearn.pipeline.make_pipeline(decomposition.PCA(n_components=2), naive_bayes.GaussianNB())
unscaled_clf = pipeline.make_pipeline(decomposition.PCA(n_components=2),
                                      naive_bayes.GaussianNB())

unscaled_clf.fit(Xtrain, ytrain)
pred_test = unscaled_clf.predict(Xtest)

# pour les données mises à l'échelle
# std_clf = sklearn.pipeline.make_pipeline(preprocessing.StandardScaler(), decomposition.PCA(n_components=2), naive_bayes.GaussianNB())
std_clf = pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                 decomposition.PCA(n_components=2),
                                 naive_bayes.GaussianNB())
std_clf.fit(Xtrain, ytrain)
pred_test_std = std_clf.predict(Xtest)


# In[72]:


# Apprentissage non-supervisé (ACP/PCA)

import seaborn as sns
sns.set()

X_database = df_copy.drop('Statut', axis=1)
y_database = df_copy['Statut']


# étape 1
from sklearn.decomposition import PCA

# étape 2
model = PCA(n_components=1)

# étape 4
model.fit(X_database)

# étape 5
X_2D = model.transform(X_database)

df['PCA1'] = X_2D[:, ]
df['PCA2'] = X_2D[:, ]


# In[73]:


sns.lmplot(x = "PCA1", y = "PCA2", hue='Statut', data=df, fit_reg=False);


# In[74]:


import matplotlib.pyplot as plt  
import statsmodels.api as sm  
from statsmodels.formula.api import ols 
from scipy import stats
import seaborn as sns  
import numpy as np  
import pandas.tseries  
plt.style.use('fivethirtyeight')  


# In[75]:


# Visualiser la distribution de mes données

f, ax = plt.subplots( figsize = (11,9) )  
sns.distplot(df[df.Statut == "Non-dévelopé"].Espérance, ax = ax, label = 'Pays en voie de développement')  
sns.distplot(df[df.Statut == "Dévelopé"].Espérance, ax = ax, label = 'Pays développés')  
plt.title( 'Distribution de espérance de vie selon le statut des pays')  
plt.legend()  
plt.show()  


# In[76]:


# Trouver les pays avec l'espérance de vie la plus élevée et la plus faible

for i in range(0, len(df["Espérance"])) :    # Je veux jouer dans ma colonne price
    pays_max = df["Country"][i]       # Contenu  en string (acura, audi, etc.)
    pays_min = df["Country"][i]
    esperance_max = max(df["Espérance"])
    esperance_min = min(df["Espérance"])
    if df["Espérance"][i] == max(df["Espérance"]) :
        print(f'Le pays avec l\'espérance de vie la plus élevée est le {pays_max} avec une espérance de vie de {esperance_max} années'
)
    if df["Espérance"][i] == min(df["Espérance"]) :
        print(f'Le pays avec l\'espérance de vie la plus faible est le {pays_min} avec une espérance de vie de {esperance_min} années'
)
       


# In[77]:


# Tester la normalité
print(stats.skew(df['Espérance']),
      stats.kurtosis(df['Espérance']))
# Je peux donc faire une ANOVA


# In[78]:


# Va aller chercher dans la colonne STATUT, toutes les données ESPÉRANCE des "DÉVELOPPÉS"

develop_esperance = df[df['Statut'] == 'Dévelopé']['Espérance']

# Va aller chercher dans la colonne STATUT, toutes les données ESPÉRANCE des "Non-dévelopés"

non_develop_esperance = df[df['Statut'] == 'Non-dévelopé']['Espérance']


# In[82]:


f_oneway(develop_esperance, non_develop_esperance)


# In[83]:


groupby_statut = df.groupby(["Statut"])


# In[84]:


groupby_statut.describe()


# In[85]:


import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.boxplot(x='Statut', y='Espérance', data=df, color='#99c2a2')
ax = sns.swarmplot(x="Statut", y="Espérance", data=df, color='#7d0013')
plt.show()


# In[86]:


# Algorithme automatisés (algorithme de recherche linéaire)

def linear_search1(Integer, Table):

    x = Integer # une variable que nous voulons trouver; integer, float, character, string
    Table = [a1, a2, ..., an] # une table de valeurs pouvant contenir la valeur à rechercher

    for i in Table: # lire toutes les valeurs de la table, appelons-le 'i'
        
        if i == x:

            reponse = True

    return reponse


# In[ ]:


# Implémenter sqlite (en utilisant un script pour la gestion d'erreur)

import sqlite3

try:
    sqliteConnection = sqlite3.connect('/Users/imendoukhane/Library/Mobile Documents/com~apple~Numbers/Documents/Human_life_Expectancy_lucas.xlsx')
    cursor = sqliteConnection.cursor()
    print("Database created and Successfully Connected to SQLite")

    sqlite_select_Query = "select sqlite_version();"
    cursor.execute(sqlite_select_Query)
    record = cursor.fetchall()
    print("SQLite Database Version is: ", record)
    cursor.close()

except sqlite3.Error as error:
    print("Error while connecting to sqlite", error)
finally:
    if sqliteConnection:
        sqliteConnection.close()
        print("The SQLite connection is closed")
        

