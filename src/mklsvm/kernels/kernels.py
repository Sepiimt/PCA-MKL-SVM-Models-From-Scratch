import numpy as np


class IterativeMeta(type):
    def __iter__(cls):
        return iter(cls.__subclasses__())


class Kernels(metaclass=IterativeMeta):
    def __init__(self, center=True, normalize=True):
        self.K = None
        self.center = center
        self.normalize = normalize
        self.train_diag_sqrt = None  # Caches diagonal scaling vector
        self.train_col_means = None  # Caches column means of raw training matrix
        self.train_grand_mean = None # Caches grand mean of raw training matrix


    def post_process_fit(self):
        # --- Applies Centering and/or Normalization Sequentially Based on Instance Flags ---
        if self.center:
            self._center_matrix()
        if self.normalize:
            self._normalize_matrix()

    def _center_matrix(self):
            # --- Centers the Training Kernel Matrix K in Feature Space in-Place ---
            #> Formula: K_centered = K - col_means - row_means + grand_mean
            if self.K is None:
                raise ValueError("Kernel matrix K has not been fitted yet.")
            # --- Caching Uncentered Training Statistics ---
            self.train_col_means = np.mean(self.K, axis=0, keepdims=True)  #> Shape: (1, N)
            self.train_grand_mean = np.mean(self.K)                        #> Scalar
            # --- Broadcasted Vectorized Centering ---
            self.K = self.K - self.train_col_means - self.train_col_means.T + self.train_grand_mean

    def _normalize_matrix(self):
            if self.K is None:
                raise ValueError("Kernel matrix K has not been fitted yet.")
            #> Extracting diagonal elements and safeguard against zero/negative entries
            train_diag = np.diag(self.K)
            self.train_diag_sqrt = np.sqrt(np.maximum(train_diag, 1e-12))
            #> Applying transformation: K_ij / sqrt(K_ii * K_jj)
            scaling_matrix = np.outer(self.train_diag_sqrt, self.train_diag_sqrt)
            self.K = self.K / scaling_matrix


    def post_process_cross(self, K_cross, test_diag):
        # --- Applies Cross-Centering and/or Cross-Normalization Based on Instance Flags ---
        if self.center:
            K_cross, test_diag = self._center_cross_matrix(K_cross, test_diag)
        if self.normalize:
            K_cross = self._normalize_cross_matrix(K_cross, test_diag)
        return K_cross

    def _center_cross_matrix(self, K_cross, test_diag):
        # --- Out-of-sample Feature Space Centering for Cross-Kernel Matrices ---
        if self.train_col_means is None or self.train_grand_mean is None:
            raise ValueError("Training centering statistics missing. Fit the kernel first.")
        #> Row means for test points against training set: Shape (N_test, 1)
        test_row_means = np.mean(K_cross, axis=1, keepdims=True)
        #> Out-of-sample cross matrix centering
        K_cross_centered = K_cross - self.train_col_means - test_row_means + self.train_grand_mean
        #> Out-of-sample centered diagonal: k(x*, x*) - 2 * mean(k(x*, X_train)) + grand_mean
        test_diag_centered = test_diag - 2 * test_row_means.ravel() + self.train_grand_mean
        # --- Return ---
        return K_cross_centered, test_diag_centered

    def _normalize_cross_matrix(self, K_cross, test_diag):
            # --- None Value Guard ---
            if self.train_diag_sqrt is None:
                raise ValueError("Training matrix scaling vectors are missing.")
            # --- Computation ---
            test_diag_sqrt = np.sqrt(np.maximum(test_diag, 1e-12))
            #> Outer product order ensures shape matches (N_test, N_SVs)
            scaling_matrix = np.outer(test_diag_sqrt, self.train_diag_sqrt)
            # --- Return ---
            return K_cross / scaling_matrix


    def prune_support_vectors(self, sv_indices):
        # --- Trimming Training Data ---
        #> Slices cached training statistics down to support vectors only.
        if self.train_col_means is not None:
            self.train_col_means = self.train_col_means[:, sv_indices]
        if self.train_diag_sqrt is not None:
            self.train_diag_sqrt = self.train_diag_sqrt[sv_indices]
    

    def denormalize_matrix(self):
        if self.K is None:
            raise ValueError("Kernel matrix K is empty.")
        if self.train_diag_sqrt is None:
            raise ValueError("Matrix was never normalized or missing cached scaling factors.")
        # --- Denormalizing ---
        scaling_matrix = np.outer(self.train_diag_sqrt, self.train_diag_sqrt)
        self.K = self.K * scaling_matrix


    def save_model(self, filepath):
        model_data = {
            'center': self.center,
            'normalize': self.normalize,
            'train_diag_sqrt': self.train_diag_sqrt,
            'train_col_means': self.train_col_means,
            'train_grand_mean': self.train_grand_mean
        }
        np.save(filepath, model_data)


    def load_model(self, filepath):
        model_data = np.load(filepath, allow_pickle=True).item()
        self.center = model_data.get('center', True)
        self.normalize = model_data.get('normalize', True)
        self.train_diag_sqrt = model_data.get('train_diag_sqrt', None)
        self.train_col_means = model_data.get('train_col_means', None)
        self.train_grand_mean = model_data.get('train_grand_mean', None)



class LinearKernel(Kernels):
    def fit_K(self, X, code_logic_arg=None):
        self.K = np.dot(X, X.T)
        self.post_process_fit()

    def compute_diag(self, X):
        #> O(N) evaluation of the diagonal elements
        return np.sum(X**2, axis=1)

    def compute_cross_K(self, train_X, test_X, code_logic_arg=None):
        K_cross = np.dot(test_X, train_X.T)
        test_diag = self.compute_diag(test_X)
        return self.post_process_cross(K_cross, test_diag)



class PolynomialKernel(Kernels):
    def fit_K(self, X, c_and_d_tuple):
        c, d = c_and_d_tuple[0], c_and_d_tuple[1]
        self.K = (np.dot(X, X.T) + c) ** d
        self.post_process_fit()

    def compute_diag(self, X, c_and_d_tuple):
        c, d = c_and_d_tuple[0], c_and_d_tuple[1]
        return (np.sum(X**2, axis=1) + c) ** d

    def compute_cross_K(self, train_X, test_X, c_and_d_tuple):
        c, d = c_and_d_tuple[0], c_and_d_tuple[1]
        K_cross = (np.dot(test_X, train_X.T) + c) ** d
        test_diag = self.compute_diag(test_X, c_and_d_tuple)
        return self.post_process_cross(K_cross, test_diag)



class RBFKernel(Kernels):
    def fit_K(self, X, gamma):
        sq_norms = np.sum(X**2, axis=1, keepdims=True)
        pairwise_dists = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
        self.K = np.exp(-gamma * pairwise_dists)
        self.post_process_fit()

    def compute_diag(self, X):
        # RBF diagonal elements are always 1.0
        return np.ones(len(X))

    def compute_cross_K(self, train_X, test_X, gamma):
        sq_norms1 = np.sum(test_X**2, axis=1, keepdims=True)
        sq_norms2 = np.sum(train_X**2, axis=1, keepdims=True)
        pairwise_dists = sq_norms1 + sq_norms2.T - 2 * np.dot(test_X, train_X.T)
        K_cross = np.exp(-gamma * pairwise_dists)
        test_diag = self.compute_diag(test_X)
        return self.post_process_cross(K_cross, test_diag)



class LaplacianKernel(Kernels):
    def fit_K(self, X, gamma):
        #> Pairwise L1 (Manhattan) distances via broadcasting
        pairwise_dists = np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
        self.K = np.exp(-gamma * pairwise_dists)
        self.post_process_fit()

    def compute_diag(self, X):
        # Laplacian diagonal elements are always 1.0
        return np.ones(len(X))

    def compute_cross_K(self, train_X, test_X, gamma):
        pairwise_dists = np.sum(np.abs(test_X[:, None, :] - train_X[None, :, :]), axis=2)
        K_cross = np.exp(-gamma * pairwise_dists)
        test_diag = self.compute_diag(test_X)
        return self.post_process_cross(K_cross, test_diag)



class RationalQuadraticKernel(Kernels):
    def fit_K(self, X, alpha_and_length_tuple):
        alpha, length_scale = alpha_and_length_tuple[0], alpha_and_length_tuple[1]
        sq_norms = np.sum(X**2, axis=1, keepdims=True)
        pairwise_sq_dists = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
        self.K = (1 + pairwise_sq_dists / (2 * alpha * length_scale**2)) ** (-alpha)
        self.post_process_fit()

    def compute_diag(self, X):
        # Rational quadratic diagonal elements are always 1.0 (zero self-distance)
        return np.ones(len(X))

    def compute_cross_K(self, train_X, test_X, alpha_and_length_tuple):
        alpha, length_scale = alpha_and_length_tuple[0], alpha_and_length_tuple[1]
        sq_norms1 = np.sum(test_X**2, axis=1, keepdims=True)
        sq_norms2 = np.sum(train_X**2, axis=1, keepdims=True)
        pairwise_sq_dists = sq_norms1 + sq_norms2.T - 2 * np.dot(test_X, train_X.T)
        K_cross = (1 + pairwise_sq_dists / (2 * alpha * length_scale**2)) ** (-alpha)
        test_diag = self.compute_diag(test_X)
        return self.post_process_cross(K_cross, test_diag)



class SigmoidKernel(Kernels):
    def fit_K(self, X, gamma_and_c_tuple):
        gamma, c = gamma_and_c_tuple[0], gamma_and_c_tuple[1]
        self.K = np.tanh(gamma * np.dot(X, X.T) + c)
        # --- PSD Guard ---
        min_eig = np.min(np.linalg.eigvalsh(self.K))
        if min_eig < -1e-5:
            raise ValueError(f"Sigmoid kernel is not PSD for gamma={gamma}, c={c}. Min eigenvalue: {min_eig}")
        self.post_process_fit()

    def compute_diag(self, X, gamma_and_c_tuple):
        gamma, c = gamma_and_c_tuple[0], gamma_and_c_tuple[1]
        return np.tanh(gamma * np.sum(X**2, axis=1) + c)

    def compute_cross_K(self, train_X, test_X, gamma_and_c_tuple):
        gamma, c = gamma_and_c_tuple[0], gamma_and_c_tuple[1]
        K_cross = np.tanh(gamma * np.dot(test_X, train_X.T) + c)
        test_diag = self.compute_diag(test_X, gamma_and_c_tuple)
        return self.post_process_cross(K_cross, test_diag)