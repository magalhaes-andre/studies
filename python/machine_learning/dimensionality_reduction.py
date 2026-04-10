"""Dimensionality reduction (PCA) refresher.

Dimensionality reduction compresses a dataset with many features into a smaller set of
new features while preserving as much information as possible. This script demonstrates
Principal Component Analysis (PCA) on the Iris dataset by reducing 4 measured flower
features down to 2 principal components.

Key ideas used here:
    - Standardize features first: PCA is variance-based, so scaling to zero mean / unit
    variance prevents large-scale features from dominating the components.
    - Principal components are new, orthogonal axes: each component is a linear combination
    of the original features, chosen to capture maximum remaining variance.
    - Explained variance ratio: pca.explained_variance_ratio_ tells how much of the total
    variance each component captures; the sum indicates how much information is retained
    after reduction.

Typical uses:
    - Visualization (2D/3D projections), noise reduction, faster downstream models, and
    mitigating multicollinearity.

Caveat:
    - PCA components are less interpretable than original features, and PCA captures linear
    structure (nonlinear structure may need other methods).
"""

from IPython.display import display
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['Target'] = iris.get('target')
display(df.head())

# Separation of resources into x and destination into y
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = df[features].values
y = df['Target'].values

# Start PCA analysis, escalonating data

from sklearn.preprocessing import StandardScaler

standardized_x = StandardScaler().fit_transform(x)
standardized_df = pd.DataFrame(data = standardized_x, columns = features)
display(standardized_df.head())

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_components = pca.fit_transform(standardized_x)

df_pca = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])

target = pd.Series(iris['target'], name='target')

result_df = pd.concat([df_pca, target], axis=1)
display(result_df)

print('Variance of each component:', pca.explained_variance_ratio_)
print('Total Variance explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))