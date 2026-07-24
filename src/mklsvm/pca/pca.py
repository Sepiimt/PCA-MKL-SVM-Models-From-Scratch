# --- Imports ---
import numpy as np
import pandas as pd


class PCA:
    def __init__(self):
        # --- Scaler and Descaler Values (kept separate from PCA-fit centering values below) ---
        self.scale_means = None
        self.scale_stds = None
        # --- Eigendecomposition (SVD) Values ---
        self.eigenvalues = None
        self.eigenvectors = None
        self.means = None
        # --- Original Data Info ---
        self.orig_features = None


    def scaler(self, X, fit=True):
        #> fit=True: compute means/stds from X (use on train), store them, and scale X.
        #> fit=False: reuse previously stored means/stds (use on test/new data) - does not recompute or overwrite them.
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
        if fit:
            means = X.mean(axis=0)
            stds = X.std(axis=0)
            self.sub_scaler_value_updater(means, stds)
        else:
            if self.scale_means is None or self.scale_stds is None:
                raise ValueError("Error: Scaler has not been fit yet. Call model.scaler(X, fit=True) first!")
            means = self.scale_means
            stds = self.scale_stds
        scaled_X = (X - means) / stds
        # --- Return ---
        return scaled_X
    
    def sub_scaler_value_updater(self, means, stds):
        self.scale_means = means
        self.scale_stds = stds


    def descaler(self, X):
        if self.scale_means is None or self.scale_stds is None:
            raise ValueError("Error: Scaler values have not been saved. Use model.scaler() first!")
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
        means = self.scale_means.values if hasattr(self.scale_means, 'values') else self.scale_means
        stds = self.scale_stds.values if hasattr(self.scale_stds, 'values') else self.scale_stds
        descaled_X = (X * stds) + means
        # --- Return ---
        return descaled_X
    

    def svd_decomposition(self, centered_X):
        # --- SVD Direct Decomposition ---
        #> U: Left singular vectors, S: Singular values, Vt: Right singular vectors (transposed)
        _, S, Vt = np.linalg.svd(centered_X, full_matrices=False)
        # --- Converting Singular Values to Eigenvalues ---
        n_samples = centered_X.shape[0]
        eigenvalues = (S ** 2) / (n_samples - 1)
        #> Eigenvectors are the columns of V, which means rows of Vt transposed.
        eigenvectors = Vt.T
        # --- Return ---
        return eigenvalues, eigenvectors
    

    def fit(self, X, orig_features=None, reset_model_values=False):
        if reset_model_values:
            self.sub_fit_value_updater(None, None, None, None)
        # --- Extract and Center Data ---
        X_vals = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        means = X_vals.mean(axis=0)
        centered_X = X_vals - means
        # --- Decompose Using SVD ---
        eigenvalues, eigenvectors = self.svd_decomposition(centered_X)
        # --- Update Model Values ---
        self.sub_fit_value_updater(eigenvalues, eigenvectors, means, orig_features)
        # --- Returning ---
        return self.sub_fit_report()

    def sub_transform_input_validator(self, n_components):
        if self.eigenvalues is None or self.eigenvectors is None:
            raise ValueError("Error: PCA model has not been trained. Use model.fit() first!")
        if n_components < 0 or n_components > len(self.eigenvalues):
            raise ValueError("Error: `n_components` parameter has not been set correctly!")

    def sub_transform_dataframe_creator(self, X, n_components):
        n_components += 1
        X_vals = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        means_vals = self.means.values if hasattr(self.means, 'values') else self.means
        centered_X = X_vals - means_vals
        PCA_X = centered_X @ self.eigenvectors[:, :n_components]
        PCA_dataframe = pd.DataFrame(
            PCA_X,
            columns=[f"PC{i}" for i in range(n_components)])
        # --- Return ---
        return PCA_dataframe
    
    def sub_fit_value_updater(self, eigenvalues, eigenvectors, means, orig_features):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.means = means
        self.orig_features = orig_features

    def sub_fit_report(self):
        eigenvalue_sum = np.sum(self.eigenvalues)
        explained_variance_ratio = self.eigenvalues / eigenvalue_sum
        cumulative_variance = np.cumsum(explained_variance_ratio)
        result_dict = {
            'PCA': [f"PCA{i}" for i in range(len(self.eigenvalues))],
            'Explained Variance Ratio': explained_variance_ratio,
            'Cumulative Variance': cumulative_variance}
        # --- Return ---
        return pd.DataFrame(result_dict, index=None)
    

    def transform(self, X, n_components):
        self.sub_transform_input_validator(n_components)
        return self.sub_transform_dataframe_creator(X, n_components)


    def revert(self, PCA_X, n_components):
        PCA_X_vals = PCA_X.values if isinstance(PCA_X, pd.DataFrame) else np.array(PCA_X)
        reverted_X = PCA_X_vals @ self.eigenvectors[:, :n_components].T
        means_vals = self.means.values if hasattr(self.means, 'values') else self.means
        reverted_X += means_vals
        reverted_X_dataframe = pd.DataFrame(reverted_X)
        if self.orig_features is not None:
            reverted_X_dataframe.columns = self.orig_features
        # --- Return ---
        return reverted_X_dataframe
    

    def save_info(self, filepath):
        info_dict = {
            'scale_means': self.scale_means,
            'scale_stds': self.scale_stds,
            'means': self.means,
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'orig_features': self.orig_features
        }
        np.savez(filepath, **info_dict)


    def load_info(self, filepath):
        loaded_data = np.load(filepath, allow_pickle=True)
        self.scale_means = loaded_data['scale_means']
        self.scale_stds = loaded_data['scale_stds']
        self.means = loaded_data['means']
        self.eigenvalues = loaded_data['eigenvalues']
        self.eigenvectors = loaded_data['eigenvectors']
        self.orig_features = loaded_data['orig_features']