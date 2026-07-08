import numpy as np


class IterativeMeta(type):
    def __iter__(cls):
        return iter(cls.__subclasses__())


class Kernels(metaclass=IterativeMeta):
    def __init__(self):
        self.K = None
        self.train_diag_sqrt = None  # Caches scaling vector for cross-evaluation and reversion

    def normalize_matrix(self):
        if self.K is None:
            raise ValueError("Kernel matrix K has not been fitted yet.")
        #> Extracting diagonal elements and safeguard against zero/negative entries
        train_diag = np.diag(self.K)
        self.train_diag_sqrt = np.sqrt(np.maximum(train_diag, 1e-12))
        #> Applying transformation: K_ij / sqrt(K_ii * K_jj)
        scaling_matrix = np.outer(self.train_diag_sqrt, self.train_diag_sqrt)
        self.K = self.K / scaling_matrix

    def denormalize_matrix(self):
        if self.K is None:
            raise ValueError("Kernel matrix K is empty.")
        if self.train_diag_sqrt is None:
            raise ValueError("Matrix was never normalized or missing cached scaling factors.")
        # --- Denormalizing ---
        scaling_matrix = np.outer(self.train_diag_sqrt, self.train_diag_sqrt)
        self.K = self.K * scaling_matrix

    def normalize_cross_matrix(self, K_cross, test_diag):
        if self.train_diag_sqrt is None:
            raise ValueError("Training matrix scaling vectors are missing. Fit the kernel first.")
        
        test_diag_sqrt = np.sqrt(np.maximum(test_diag, 1e-12))
        scaling_matrix = np.outer(self.train_diag_sqrt, test_diag_sqrt)
        return K_cross / scaling_matrix


class LinearKernel(Kernels):
    def fit_K(self, X, code_logic_arg):
        self.K = np.dot(X, X.T)
        self.normalize_matrix()
    
    def compute_diag(self, X, code_logic_arg):
        #> O(N) evaluation of the diagonal elements
        return np.sum(X**2, axis=1)

    def compute_cross_K(self, train_X, test_X, code_logic_arg):
        K_cross = np.dot(train_X, test_X.T)
        test_diag = self.compute_diag(test_X, code_logic_arg)
        return self.normalize_cross_matrix(K_cross, test_diag)

    def save_model(self, filepath):
        model_data = {'K': self.K, 'train_diag_sqrt': self.train_diag_sqrt}
        np.save(filepath, model_data)

    def load_model(self, filepath):
        model_data = np.load(filepath, allow_pickle=True).item()
        self.K = model_data['K']
        self.train_diag_sqrt = model_data['train_diag_sqrt']


class PolynomialKernel(Kernels):
    def fit_K(self, X, c_and_d_tuple):
        c, d = c_and_d_tuple[0], c_and_d_tuple[1]
        self.K = (np.dot(X, X.T) + c) ** d
        self.normalize_matrix()

    def compute_diag(self, X, c_and_d_tuple):
        c, d = c_and_d_tuple[0], c_and_d_tuple[1]
        return (np.sum(X**2, axis=1) + c) ** d

    def compute_cross_K(self, train_X, test_X, c_and_d_tuple):
        c, d = c_and_d_tuple[0], c_and_d_tuple[1]
        K_cross = (np.dot(train_X, test_X.T) + c) ** d
        test_diag = self.compute_diag(test_X, c_and_d_tuple)
        return self.normalize_cross_matrix(K_cross, test_diag)

    def save_model(self, filepath):
        model_data = {'K': self.K, 'train_diag_sqrt': self.train_diag_sqrt}
        np.save(filepath, model_data)

    def load_model(self, filepath):
        model_data = np.load(filepath, allow_pickle=True).item()
        self.K = model_data['K']
        self.train_diag_sqrt = model_data['train_diag_sqrt']


class RBFKernel(Kernels):
    def fit_K(self, X, gamma):
        sq_norms = np.sum(X**2, axis=1, keepdims=True)
        pairwise_dists = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
        self.K = np.exp(-gamma * pairwise_dists)
        self.normalize_matrix()  # Diagonals are naturally 1.0, but enforces uniform structural state

    def compute_diag(self, X, gamma):
        # RBF diagonal elements are always 1.0
        return np.ones(len(X))

    def compute_cross_K(self, train_X, test_X, gamma):
        sq_norms1 = np.sum(train_X**2, axis=1, keepdims=True)
        sq_norms2 = np.sum(test_X**2, axis=1, keepdims=True)
        pairwise_dists = sq_norms1 + sq_norms2.T - 2 * np.dot(train_X, test_X.T)
        K_cross = np.exp(-gamma * pairwise_dists)
        test_diag = self.compute_diag(test_X, gamma)
        return self.normalize_cross_matrix(K_cross, test_diag)

    def save_model(self, filepath):
        model_data = {'K': self.K, 'train_diag_sqrt': self.train_diag_sqrt}
        np.save(filepath, model_data)

    def load_model(self, filepath):
        model_data = np.load(filepath, allow_pickle=True).item()
        self.K = model_data['K']
        self.train_diag_sqrt = model_data['train_diag_sqrt']


class SigmoidKernel(Kernels):
    def fit_K(self, X, gamma_and_c_tuple):
        gamma, c = gamma_and_c_tuple[0], gamma_and_c_tuple[1]
        self.K = np.tanh(gamma * np.dot(X, X.T) + c)
        self.normalize_matrix()

    def compute_diag(self, X, gamma_and_c_tuple):
        gamma, c = gamma_and_c_tuple[0], gamma_and_c_tuple[1]
        return np.tanh(gamma * np.sum(X**2, axis=1) + c)

    def compute_cross_K(self, train_X, test_X, gamma_and_c_tuple):
        gamma, c = gamma_and_c_tuple[0], gamma_and_c_tuple[1]
        K_cross = np.tanh(gamma * np.dot(train_X, test_X.T) + c)
        test_diag = self.compute_diag(test_X, gamma_and_c_tuple)
        return self.normalize_cross_matrix(K_cross, test_diag)
    
    def save_model(self, filepath):
        model_data = {'K': self.K, 'train_diag_sqrt': self.train_diag_sqrt}
        np.save(filepath, model_data)

    def load_model(self, filepath):
        model_data = np.load(filepath, allow_pickle=True).item()
        self.K = model_data['K']
        self.train_diag_sqrt = model_data['train_diag_sqrt']