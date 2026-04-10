"""
GrassHopper and BrushCrickets Classification
    A scientist collected sample data from two insect types
    grasshoppers and brushcrickets to find reliable differences
    between them using physical measurements. The analysis suggests
    that abdomen size and antenna length are strong indicators for
    distinguishing the species, and the goal is to automatically
    and accurately classify each insect based on these features.
    In this Jupyter notebook, we use machine learning to learn the
    patterns in the data and build a model that predicts whether an
    insect is a grasshopper or a katydid from its abdomen and antenna measurements.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
portuguese_df = pd.read_excel("python/machine_learning/classification_model_/gaf_esp.xlsx")

translation_map = {
    "Espécie": "species",
    "Comprimento do Abdômen": "abdomen_length",
    "Comprimento das Antenas": "antenna_length",
    "Gafanhoto": "Grasshopper",
    "Esperança": "Katydid"
}

df = portuguese_df.rename(columns=translation_map)
df = df.replace(translation_map)

df.groupby('species').describe()

abdomen_antenna_and_species_plot = df.plot.scatter(x='abdomen_length', y='antenna_length', c='species')
plt.show()

x = df[['abdomen_length', 'antenna_length']]
y = df['species']
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Now to train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

prediction = knn.predict([[8,6]])
print(prediction)

new_prediction = knn.predict(x_test)

accuracy = accuracy_score(y_true= y_test, y_pred= new_prediction)
print(accuracy)
