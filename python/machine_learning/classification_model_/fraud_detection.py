import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plot
import seaborn as sns
import numpy as np
import warnings

df = pd.read_csv("python/machine_learning/classification_model_/card_transdata.csv")
df.describe()