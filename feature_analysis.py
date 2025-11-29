from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np

def correlation_analysis(df):
    corr_matrix = df.corr()

    plt.figure(figsize=(6, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de correlación de variables estandarizadas")
    plt.show()

def pca_analysis(df,n_components = 2):
    # Apply PCA to the scaled data
    pca = PCA()
    pca.fit(df)

    # Analyze explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained variance ratio by each component:", explained_variance_ratio)

    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), marker='o', linestyle='--')
    plt.title('Explained Variance by Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    # Select the first two principal components
    pca_2 = PCA(n_components=n_components)
    principal_components = pca_2.fit_transform(df)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    print("\nFirst two principal components:")
    print(pca_df)

    # Interpret the loadings (eigenvectors)
    loadings = pd.DataFrame(pca_2.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=df.columns)
    print("\nLoadings (Contribution of each feature to each PC):")
    print(loadings)

    # Create a biplot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'])

    plt.title('PCA Biplot (PC1 vs PC2)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()