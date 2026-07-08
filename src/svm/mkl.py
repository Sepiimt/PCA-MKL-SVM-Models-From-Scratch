import numpy as np
from kernels import *
from smo import SMO
from utils import Timer

class MKL:
    def __init__(self):
        self.kernel_array = None
        self.kernels_arguments = None
        self.Timer = Timer()
        self.beta_array = None
        self.SMO_model = SMO()
        self.selected_kernels = None
        self.train_X = None
        self.train_Y = None

    def sub_select_kernels_beta_and_kernel_array_creator(self, selected_kernels):
        # --- Initializing Beta Values ---
        default_beta = 1 / np.sum(selected_kernels)
        beta_array = np.ones(sum(selected_kernels)) * default_beta
        kernel_array = []
        # --- Initializing Different Kernels ---
        for kernel, criteria in zip(Kernels, selected_kernels):
            if criteria is True:
                kernel_instance = kernel()
                kernel_array.append(kernel_instance)
        # --- Return ---
        return beta_array, kernel_array

    def select_kernels(self, selected_kernels=[True,True,True,True]):
        # --- Initializing Beta & Kernel Vectors ---
        self.beta_array, self.kernel_array = self.sub_select_kernels_beta_and_kernel_array_creator(selected_kernels)
        self.selected_kernels = selected_kernels

    def sub_fit_kernel_selected_checker(self):
        # --- Beta & Kernel Value Existance Checker ---
        if (self.kernel_array is None) or (self.beta_array is None):
            raise ValueError("ERROR: Please use MKL.select_kernels() first!")
        
    def sub_kernels_matrices_fit_value_yielder(self, kernels_arguments):
        #> kernels_arguments = [c,d, gamma, gamma,c)]
        # --- If Linear Kernel ---
        if self.selected_kernels[0]:
            yield "code_logic_arg" #> A dummy arguemnt for consistant input logic
        # --- If Polynomial Kernel ---
        if self.selected_kernels[1]:
            yield (kernels_arguments[0], kernels_arguments[1])
        # --- If RBF Kernel ---
        if self.selected_kernels[2]:
            yield kernels_arguments[2]
        # --- If Sigmoid Kernel ---
        if self.selected_kernels[3]:
            yield (kernels_arguments[3], kernels_arguments[4])

    def sub_fit_kernels_matrices_fit(self, X, kernel_array, kernels_arguments):
        # --- Fitting Each Kernel & Creating Kernel Matrices ---
        for kernel, argument in zip(kernel_array, self.sub_kernels_matrices_fit_value_yielder(kernels_arguments)):
            kernel.fit_K(X, argument)

    def sub_fit_create_sum_of_kernel_matrices(self, beta_array, kernel_array, len_x):
        # --- Calculating Weighted Summation of Kernel Matrices ---
        sum_of_kernel_matrices = np.zeros((len_x,len_x))
        for beta, kernel in zip(beta_array, kernel_array):
            sum_of_kernel_matrices += beta * kernel.K
        # --- Return ---
        return sum_of_kernel_matrices

    def sub_fit_SMO_fit(self, kernel_matrix, Y, smo_C, max_SMO_iter, SMO_tolerance, detailed_timer):
        # --- Training SMO Model ---
        self.SMO_model.fit(kernel_matrix, Y, C = smo_C, max_iter = max_SMO_iter, tolerance = SMO_tolerance, detailed_timer = detailed_timer)

    def sub_MKL_optimizer_alignment_score_computer(self, Y, kernel_array):
        # --- Initializing Values ---
        kernels_alignment_score_array = []
        alpha_vector = self.SMO_model.alpha_vector
        ewm_of_alpha_and_Y_vector = np.multiply(alpha_vector, Y) #> ewm: element wise multiplication
        # --- Calculating Alignment Score for Each Kernel ---
        for kernel in kernel_array:
            alignment_score = 0.5 * (ewm_of_alpha_and_Y_vector.T @ kernel.K @ ewm_of_alpha_and_Y_vector)
            kernels_alignment_score_array.append(alignment_score)
        # --- Return ---
        return kernels_alignment_score_array

    def sub_MKL_optimizer_beta_gradient_step(self, beta_array, kernels_alignment_score_array, beta_learning_rate):
        # --- Optimizing Beta Values (Need Projection Later) ---
        new_beta_array = beta_array + beta_learning_rate * np.array(kernels_alignment_score_array)
        # --- Return ---
        return new_beta_array

    def sub_MKL_optimizer_project_simplex(self, v):
        # --- Preparing Mathematical Process Requirements ---
        descending_sorted_v = np.sort(v)[::-1] 
        cumsum_of_dsv = np.cumsum(descending_sorted_v)
        ind = np.arange(len(v)) + 1
        # --- Mathematic Process ---
        cond = descending_sorted_v - (cumsum_of_dsv - 1) / ind > 0
        rho = ind[cond][-1]  #> Internal cutoff index
        theta = (cumsum_of_dsv[cond][-1] - 1) / rho  #> Internal threshold value
        # --- Return ---
        return np.maximum(v - theta, 0)

    def sub_fit_MKL_optimizer(self, Y, beta_array, kernel_array, beta_learning_rate):
        # --- Calculating Alignment Score for Each Kernel ---
        kernels_alignment_score_array = self.sub_MKL_optimizer_alignment_score_computer(Y, kernel_array)
        # --- Optimizing Beta Values (Need Projection) ---
        new_beta_array = self.sub_MKL_optimizer_beta_gradient_step(beta_array, kernels_alignment_score_array, beta_learning_rate)
        # --- Projecting New Beta Values ---
        projected_beta_array = self.sub_MKL_optimizer_project_simplex(new_beta_array)
        # --- Return ---
        return projected_beta_array

    def sub_fit_value_updater(self, new_beta_array):
        # --- Updating Value ----
        self.beta_array = new_beta_array

    def sub_fit_save_train_data(self, X, Y, kernels_arguments):
        self.train_X = X.copy()
        self.train_Y = Y.copy()
        self.kernels_arguments = kernels_arguments.copy()

    def sub_fit_tiny_optimization_checker(self, new_beta_array, beta_optimization_tolerance):
        # --- Checking for Tiny Optimization ---
        change = np.linalg.norm(self.beta_array - new_beta_array)
        if change < beta_optimization_tolerance:
            # --- Return ---
            return True
        # --- Return ---
        return False

    def sub_fit_MKL_value_reset(self):
        self.beta_array, _ = self.sub_select_kernels_beta_and_kernel_array_creator(self.selected_kernels)
        self.SMO_model.sub_fit_SMO_value_reset()

    def fit(self, X, Y, smo_C, max_SMO_iter, SMO_tolerance, beta_learning_rate, kernels_arguments,
            max_beta_optimization_iter = 500, beta_optimization_tolerance = 1e-3, detailed_timer=False):
        # --- Value Existance Stop Guard ---
        self.sub_fit_kernel_selected_checker()
        # --- resetting MKL Model ---
        self.sub_fit_MKL_value_reset()
        # --- Starting Timer ---
        if detailed_timer:
            self.Timer.start()
        # --- Saving Training Data ---
        self.sub_fit_save_train_data(X, Y, kernels_arguments)
        # --- Computing Kernel Matrices ---
        self.sub_fit_kernels_matrices_fit(X, self.kernel_array, kernels_arguments)
        # --- Beta Optimization Loop ---
        for _ in np.arange(max_beta_optimization_iter):
            # --- Computing Summation of Kernel Matrices ---
            sum_of_kernel_matrices = self.sub_fit_create_sum_of_kernel_matrices(self.beta_array, self.kernel_array, len(X))
            # --- Computing Alpha Matrix and SMO Optimization ---
            self.sub_fit_SMO_fit(sum_of_kernel_matrices, Y, smo_C, max_SMO_iter, SMO_tolerance, detailed_timer)
            # --- Optimizing Beta ---
            new_beta_array = self.sub_fit_MKL_optimizer(Y, self.beta_array, self.kernel_array, beta_learning_rate)
            is_converged = self.sub_fit_tiny_optimization_checker(new_beta_array, beta_optimization_tolerance)
            # --- Updating Value ---
            self.sub_fit_value_updater(new_beta_array)
            # --- Tiny Optimization Stop Guard ---
            if is_converged:
                break
            
        # --- Stopping Timer ---
        if detailed_timer:
            self.Timer.stop()
            print(f"MKL.fit() Elapsed Time: {self.Timer.elapsed_time()}")

    def predict(self, test_X):
        # --- Initializing Values ---
        n_train = len(self.train_X)
        n_test = len(test_X)
        K_test_combined = np.zeros((n_train, n_test))
        # --- Preparing Argument Parser ---
        args_generator = self.sub_kernels_matrices_fit_value_yielder(self.kernels_arguments)
        # --- Computing Combination of Kernel Matrices ---
        for beta, kernel, argument in zip(self.beta_array, self.kernel_array, args_generator):
            # Each kernel computes cross evaluation: X_train (rows) against test_X (columns)
            K_m = kernel.compute_cross_K(self.train_X, test_X, argument)
            K_test_combined += beta * K_m
        # --- Return ---
        #> Passing the single precomputed combined matrix down to the SMO solver layer
        return self.SMO_model.predict(K_test_combined)
    
    def save_model(self, filepath):
        # --- Saving Model ---
        model_data = {
            'beta_array': self.beta_array,
            'kernels_arguments': self.kernels_arguments,
            'selected_kernels': self.selected_kernels,
            'train_X': self.train_X,
            'train_Y': self.train_Y,
            'SMO_model': self.SMO_model}
        # --- SNO Save ---
        self.SMO_model.save_model(filepath)  #> Save the SMO model separately
        np.save(filepath + '_MKL_layer', model_data)
        # --- Kernel Save ---
        for i, kernel in enumerate(self.kernel_array):
            kernel.save_model(filepath + f'_{i}_kernel')  #> Save each kernel separately

    def load_model(self, filepath):
        # --- Loading Model ---
        model_data = np.load(filepath + '_MKL_layer.npy', allow_pickle=True).item()
        self.beta_array = model_data['beta_array']
        self.kernels_arguments = model_data['kernels_arguments']
        self.selected_kernels = model_data['selected_kernels']
        self.train_X = model_data['train_X']
        self.train_Y = model_data['train_Y']
        # --- Loading SMO ---
        self.SMO_model = SMO()
        self.SMO_model.load_model(filepath)  #> Load the SMO model separately
        # --- Loading Kernels ---
        self.kernel_array = []
        self.select_kernels(self.selected_kernels)  #> Reinitialize kernel_array based on selected_kernels
        for kernel,i in zip(self.kernel_array, range(len(self.kernel_array))):
            kernel.load_model(filepath + f'_{i}_kernel.npy')  #> Load each kernel separately
            self.kernel_array.append(kernel)