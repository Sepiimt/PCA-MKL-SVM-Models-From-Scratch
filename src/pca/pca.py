# --- Imports ---
import numpy as np
import pandas as pd

class PCA:
    def __init__(self):
        # --- Scaler and Descaler Values ---
        self.means = None
        self.stds = None
        # --- Eigendecomposition Values ---
        self.eigenvalues = None
        self.eigenvectors = None
        # --- Dataset Columns Mean Values ---
        self.means = None
        # --- Original Data Info ---
        self.orig_features = None

    def sub_scaler_value_updater(self, means, stds):
        # --- Scaler Value Updater ---
        self.means = means
        self.stds = stds

    def scaler(self, X, save_scaler_values=True):
        # --- Extract Values Safely ---
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
        # --- Scaling ---
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        scaled_X = (X - means) / stds
        # --- Save Scaler Values ---
        if save_scaler_values:
            self.sub_scaler_value_updater(means, stds)
        # --- Return ---
        return scaled_X

    def descaler(self, X):
        # --- Stop Guard ---
        if self.means is None or self.stds is None:
            raise ValueError("Scaler values have not been saved. Use model.scaler() first.")
        # --- Extract Values Safely ---
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
        # --- Descaling ---
        means = self.means.values if hasattr(self.means, 'values') else self.means
        stds = self.stds.values if hasattr(self.stds, 'values') else self.stds
        descaled_X = (X * stds) + means
        # --- Return ---
        return descaled_X

    def covariance_matrix(self, X):
        #--- Covariance Matrix ---
        #> Note that np.cov centers the data by default.
        return np.cov(X, rowvar=False) #> Cov(X) = E[(X - μ)(X - μ)^T]
    
    def eigen_decomposition(self, cov_matrix):
        # --- Eigen Decomposition ---
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) #> Ax = λx
        # --- Sorting ---
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # --- Return ---
        return eigenvalues, eigenvectors
    
    def sub_fit_value_updater(self, eigenvalues, eigenvectors, means, orig_features):
        # --- Fit Value Updater ---
        #> Update the fit values with the eigenvalues and eigenvectors
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.means = means
        self.orig_features = orig_features

    def sub_fit_report(self):
        # --- Calculating Data ---
        eigenvalue_sum = np.sum(self.eigenvalues)
        explained_variance_ratio = self.eigenvalues / eigenvalue_sum
        cumulative_variance = np.cumsum(explained_variance_ratio)
        # --- Building Result Dataframe ---
        result_dict = {
            'PCA':[f"PCA{i}" for i in range(len(self.eigenvalues))],
            'Explained Variance Ratio':explained_variance_ratio,
            'Cumulative Variance':cumulative_variance}
        # --- Return ---
        return pd.DataFrame(result_dict, index=None)

    def fit(self, X, orig_features=None, reset_model_values=False):
        # --- Resetting Fit Values ---
        if reset_model_values:
            self.sub_fit_value_updater(None,None,None,None)
        # --- Calculating Covariance Matrix ---
        cov_matrix = self.covariance_matrix(X)
        # --- Calculating Eigenvalues and Eigenvectors ---
        eigenvalues, eigenvectors = self.eigen_decomposition(cov_matrix)
        # --- Calculating Means for Later ---
        means = X.mean(axis=0)
        # --- Updating Fit Values ---
        self.sub_fit_value_updater(eigenvalues, eigenvectors, means, orig_features)
        # --- Returning Fit Report ---
        return self.sub_fit_report()

    def sub_transform_input_validator(self,n_components):
        # --- Check if Fit Values are Updated ---
        if self.eigenvalues is None or self.eigenvectors is None:
            raise ValueError("PCA model has not been trained. Use model.fit() first.")
        # --- Check if n_components is Valid ---
        if n_components <= 0 or n_components > len(self.eigenvalues):
            raise ValueError("`n_components` parameter has not been set correctly")

    def sub_transform_dataframe_creator(self, X, n_components):
        # --- Extract Values Safely ---
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        means_vals = self.means.values if hasattr(self.means, 'values') else self.means
        # --- Centering the Data ---
        centered_X = X_vals - means_vals
        # --- Computing PCA Columns ---
        PCA_X = centered_X @ self.eigenvectors[:,:n_components]
        # --- Preventing Pandas From Attempting to Reindex ---
        X_values = PCA_X.values if isinstance(PCA_X, pd.DataFrame) else PCA_X
        # --- Building PCA Dataframe ---
        PCA_dataframe = pd.DataFrame(
            X_values,
            columns = [f"PC{i+1}" for i in range(n_components)])
        # --- Return ---
        return PCA_dataframe

    def transform(self, X, n_components):
        # --- Input Validation ---
        self.sub_transform_input_validator(n_components)
        # --- Creating PCA Dataframe ---
        dataframe = self.sub_transform_dataframe_creator(X, n_components)
        # --- Return ---
        return dataframe

    def revert(self, PCA_X, n_components):
        # --- Extract Values Safely ---
        PCA_X_vals = PCA_X.values if isinstance(PCA_X, pd.DataFrame) else PCA_X
        # --- Reverting PCA Dataset ---
        reverted_X = PCA_X_vals @ self.eigenvectors[:,:n_components].T
        means_vals = self.means.values if hasattr(self.means, 'values') else self.means
        reverted_X += means_vals
        # --- Transforming to Dataframe ---
        reverted_X_dataframe = pd.DataFrame(reverted_X)
        # --- Trying to Set Column Names ---
        if self.orig_features is not None:
            reverted_X_dataframe.columns = self.orig_features
        # --- Return ---
        return reverted_X_dataframe
    
    def save_info(self, filepath):
        # --- Saving PCA Model Info ---
        info_dict = {
            'means': self.means,
            'stds': self.stds,
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'orig_features': self.orig_features
        }
        np.savez(filepath, **info_dict)

    def load_info(self, filepath):
        # --- Loading PCA Model Info ---
        loaded_data = np.load(filepath, allow_pickle=True)
        self.means = loaded_data['means']
        self.stds = loaded_data['stds']
        self.eigenvalues = loaded_data['eigenvalues']
        self.eigenvectors = loaded_data['eigenvectors']
        self.orig_features = loaded_data['orig_features']