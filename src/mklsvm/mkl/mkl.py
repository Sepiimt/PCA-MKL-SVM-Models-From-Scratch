import numpy as np
from ..kernels import Kernels
from ..smo import SMO
from ..utils import timer
from .optimizers import *


class MKL:
    def __init__(self):
        # --- Kernel Related ---
        self.selected_kernels = None #> Array of selected kernels with boolian values
        self.kernels_instances_matrix = None #> Same lenght as "self.selected_kernels", filled each kernel's k matrix
        self.kernels_arguments = None #> Kernel's argument in specific order
        self.kernel_centering = None
        self.kernel_normalization = None
        # --- MKL Related ---
        self.beta_array = None #> MKL's betta array
        self.selected_bom = None #> Selected "Beta Optimization Method" (BOM) instance
        self.available_bom = {"SimpleMKL": ReducedGradientOptimizer(),
                              "L2-MKL": AnalyticL2Optimizer(),
                              "Entropic Descent": ExponentiatedGradientOptimizer(),
                              "Hybrid L1/L2": ProximalHybridOptimizer(),
                              "KTA": KernelTargetAlignmentOptimizer(),
                              "Frank-Wolfe": FrankWolfeOptimizer(),
                              "FISTA": AcceleratedProjectedGradientOptimizer()}
        self.kernels = np.array(["Linear","Polynomial","RBF","Laplacian","Rational Quadratic","Sigmoid"])
        # --- Train Data ---
        self.train_X = None
        self.train_Y = None
        self.support_X = None #> Support Venctor's X indexes
        self.support_Y = None #> Support Venctor's Y indexes
        # --- SMO Related ---
        self.SMO_model = SMO()
        self.cache_initial_alpha = None #> SMO's alpha vector cache
        self.cache_initial_b = None #> SMO's beta cache
        # --- Utils ---
        self.timer = timer()


    def select_kernels(self, selected_kernels, centering, normalization):
        # --- Initializing Beta & Kernel Vectors ---
        self.beta_array, self.kernels_instances_matrix = self._beta_and_kernels_instances_matrix_creator(selected_kernels, 
                                                                                                         centering = centering, 
                                                                                                         normalization = normalization)
        # --- Saving Flags ---
        self.kernel_centering = centering
        self.kernel_normalization = normalization
        # --- Saving Selected Kernels ---
        self.selected_kernels = selected_kernels

    def _beta_and_kernels_instances_matrix_creator(self, selected_kernels, centering, normalization):
        # --- Division by Zero Guard ---
        if np.sum(selected_kernels) == 0:
            raise ValueError("ERROR: At least one kernel must be selected!")
        # --- Initializing Beta Values ---
        default_beta = 1 / np.sum(selected_kernels)
        beta_array = np.ones(sum(selected_kernels)) * default_beta
        kernels_instances_matrix = []
        # --- Initializing Different Kernels ---
        for kernel, criteria in zip(Kernels, selected_kernels):
            if criteria is True:
                kernel_instance = kernel(center = centering, normalize = normalization)
                kernels_instances_matrix.append(kernel_instance)
        # --- Return ---
        return beta_array, kernels_instances_matrix
        

    def kernels_value_yielder(self, kernels_arguments):
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
        # --- if Laplacian Kernel ---
        if self.selected_kernels[3]:
            yield kernels_arguments[3]
        # --- if Rational Quadratic Kernel ---
        if self.selected_kernels[4]:
            yield (kernels_arguments[4], kernels_arguments[5])
        # --- If Sigmoid Kernel ---
        if self.selected_kernels[5]:
            yield (kernels_arguments[6], kernels_arguments[7])


    def kernels_instances_matrix_fit(self, X, kernels_instances_matrix, kernels_arguments):
        # --- Fitting Each Kernel & Creating Kernel Matrices ---
        for kernel, argument in zip(kernels_instances_matrix, self.kernels_value_yielder(kernels_arguments)):
            kernel.fit_K(X, argument)


    def create_sum_of_kernels_instances_matrix(self, beta_array, kernels_instances_matrix, len_x):
        # # --- Calculating Weighted Summation of Kernel Matrices ---
        # sum_of_kernels_k_matrix = np.zeros((len_x,len_x))
        # for beta, kernel in zip(beta_array, kernels_instances_matrix):
        #     sum_of_kernels_k_matrix += beta * kernel.K

        stack = np.stack([kernel.K for kernel in kernels_instances_matrix])
        combined = np.tensordot(beta_array, stack, axes=1)

        # --- Return ---
        return combined


    def smo_model_fit(self, kernel_matrix, Y, smo_c, max_smo_iter, smo_tolerance, initial_alpha, initial_b, detailed_info):
        # --- Training SMO Model ---
        self.SMO_model.fit(kernel_matrix, Y, C = smo_c, max_iter = max_smo_iter, tolerance = smo_tolerance, 
                           initial_alpha = initial_alpha, initial_b = initial_b, detailed_info = detailed_info)
        

    def mkl_bom_get_instance(self, beta_optimization_method):
        return self.available_bom[beta_optimization_method]
    

    def mkl_beta_optimizer(self, Y, kernels_instances_matrix, beta_array):
        new_beta_array = self.selected_bom.update(self.SMO_model.alpha_vector, Y, kernels_instances_matrix, beta_array)
        # --- Return ---
        return new_beta_array
    

    def beta_convergence_checker(self, new_beta_array, bo_tolerance):
        # --- Checking for Tiny Optimization ---
        beta_change = np.linalg.norm(new_beta_array - self.beta_array , ord=np.inf)
        # --- Checking for Convergence ---
        if beta_change < bo_tolerance:
            # --- Return ---
            return True
        # --- Return ---
        return False


    def fit(self, X, Y, 
            smo_c, 
            max_smo_iter, 
            smo_tolerance, 
            kernels_arguments,
            beta_optimization_method = "Entropic Descent",
            max_bo_iter = 500,
            bo_tolerance = 1e-3, 
            detailed_info=False):
        # --- Value Existance Stop Guard ---
        self._kernel_selected_checker()
        # --- resetting MKL Model ---
        self._mkl_value_reset()
        # --- Starting timer ---
        if detailed_info:
            self.timer.start()
        # --- Saving Training Data ---
        self._save_train_data(X, Y, kernels_arguments, beta_optimization_method)
        # --- Computing Kernel Matrices ---
        self.kernels_instances_matrix_fit(X, self.kernels_instances_matrix, kernels_arguments)
        # --- Detailed Info ---
        if detailed_info:
            print(f"Active Kernels: {self.kernels[np.array(self.selected_kernels)]}")
            print(f"MKL.fit() Init Beta Array: {self.beta_array}\n")
        # --- Determining the Approach ---
        if beta_optimization_method == "KTA":
            iteration = self._none_smo_based_beta_optimization(X, Y,
                                                               self.kernels_instances_matrix, self.beta_array,
                                                               smo_c, max_smo_iter, smo_tolerance,
                                                               detailed_info)
        else:
            iteration = self._smo_based_beta_optimization(X, Y,
                                                          self.kernels_instances_matrix, self.beta_array,
                                                          max_bo_iter, bo_tolerance,
                                                          smo_c, max_smo_iter, smo_tolerance,
                                                          detailed_info)
        # --- Extracting Support Vectors ---
        self.SMO_model.extract_support_vectors(Y, smo_tolerance)
        sv_indices = self._extract_support_vectors(return_sv=True)
        # --- Stopping timer ---
        if detailed_info:
            self.timer.stop()
            print(f"MKL.fit() Elapsed Time Is {self.timer.elapsed_time()} On {iteration} Iteration.\n")
            print(f"Dataset's Length: {len(X)}")
            print(f"Model's Support Vector Count: {len(sv_indices)}")
            print(f"MKL.fit() Final Beta Array: {self.beta_array}")

    def _kernel_selected_checker(self):
        # --- Beta & Kernel Value Existance Checker ---
        if (self.kernels_instances_matrix is None) or (self.beta_array is None):
            raise ValueError("ERROR: Please use MKL.select_kernels() first!")

    def _value_updater(self, new_beta_array):
        # --- Updating Value ----
        self.beta_array = new_beta_array
        self.cache_initial_alpha = self.SMO_model.alpha_vector
        self.cache_initial_b = self.SMO_model.b

    def _save_train_data(self, X, Y, kernels_arguments, beta_optimization_method):
        self.train_X = X
        self.train_Y = Y
        self.kernels_arguments = kernels_arguments
        self.selected_bom = self.mkl_bom_get_instance(beta_optimization_method)

    def _mkl_value_reset(self):
        self.beta_array, _ = self._beta_and_kernels_instances_matrix_creator(self.selected_kernels, 
                                                                             self.kernel_centering, 
                                                                             self.kernel_normalization)
        self.SMO_model._value_reset()
        self.cache_initial_alpha = None
        self.cache_initial_b = None

    def _extract_support_vectors(self, return_sv=False):
        # --- Inheriting Sparsity from SMO ---
        sv_indices = self.SMO_model.sv_indices
        # --- Slicing MKL's Stored Data to SVs Only ---
        self.support_X = self.train_X[sv_indices]
        self.support_Y = self.train_Y[sv_indices]
        # --- Synchronizing Kernel State ---
        #> Slicing the scaling vectors to match the extracted support vectors
        for kernel in self.kernels_instances_matrix:
            kernel.prune_support_vectors(sv_indices)
        # --- Cleaning Up ---
        for kernel in self.kernels_instances_matrix:
            kernel.K = None
        self.train_X, self.train_Y = None, None
        # --- Returning ---
        if return_sv is True:
            return sv_indices

    def _smo_based_beta_optimization(self, X, Y,
                                     kernels_instances_matrix, beta_array,
                                     max_bo_iter, bo_tolerance,
                                     smo_c, max_smo_iter, smo_tolerance,
                                     detailed_info):
        # --- Beta Optimization Loop ---
        for iteration in np.arange(max_bo_iter):
            # --- Computing Summation of Kernel Matrices ---
            sum_of_kernels_k_matrix = self.create_sum_of_kernels_instances_matrix(beta_array, kernels_instances_matrix, len(X))
            # --- Computing Alpha Matrix and SMO Optimization ---
            self.smo_model_fit(sum_of_kernels_k_matrix, Y, smo_c, max_smo_iter, smo_tolerance, 
                                 self.cache_initial_alpha, self.cache_initial_b, detailed_info)
            # --- Optimizing Beta ---
            new_beta_array = self.mkl_beta_optimizer(Y, kernels_instances_matrix, beta_array)
            # --- Checking for Convergence ---
            is_converged = self.beta_convergence_checker(new_beta_array, bo_tolerance)
            # --- Updating Value ---
            self._value_updater(new_beta_array)
            # --- Detailed Info ---
            if detailed_info:
                print(f"MKL.fit() New Beta Array: {self.beta_array}\n")
            # --- Tiny Optimization Stop Guard ---
            if is_converged:
                break
        # --- Computing Summation of Kernel Matrices ---
        sum_of_kernels_k_matrix = self.create_sum_of_kernels_instances_matrix(self.beta_array, self.kernels_instances_matrix, len(X))
        # --- Computing Alpha Matrix and SMO Optimization ---
        self.smo_model_fit(sum_of_kernels_k_matrix, Y, smo_c, max_smo_iter, smo_tolerance, 
                                 self.cache_initial_alpha, self.cache_initial_b, detailed_info)
        # --- Return ---
        return iteration + 1
    
    def _none_smo_based_beta_optimization(self, X, Y,
                                          kernels_instances_matrix, beta_array,
                                          smo_c, max_smo_iter, smo_tolerance,
                                          detailed_info):
        # --- KTA BOM ---
        new_beta_array = self.mkl_beta_optimizer(Y, kernels_instances_matrix, beta_array)
        # --- Computing Summation of Kernel Matrices ---
        sum_of_kernels_k_matrix = self.create_sum_of_kernels_instances_matrix(new_beta_array, kernels_instances_matrix, len(X))
        # --- Computing Alpha Matrix and SMO Optimization ---
        self.smo_model_fit(sum_of_kernels_k_matrix, Y, smo_c, max_smo_iter, smo_tolerance,
                             self.cache_initial_alpha, self.cache_initial_b, detailed_info)
        # --- Updating Value ---
        self._value_updater(new_beta_array)
        # --- Detailed Info ---
        if detailed_info:
            print(f"MKL.fit() New Beta Array: {self.beta_array}\n")
        # --- Return ---
        return 1
    

    def predict(self, test_X):
        # --- Initializing Values ---
        n_train = len(self.support_X)
        n_test = len(test_X)
        K_test_combined = np.zeros((n_test, n_train))
        # --- Preparing Argument Parser ---
        args_generator = self.kernels_value_yielder(self.kernels_arguments)
        # --- Computing Combination of Kernel Matrices ---
        for beta, kernel, argument in zip(self.beta_array, self.kernels_instances_matrix, args_generator):
            # Each kernel computes cross evaluation: X_train (rows) against test_X (columns)
            K_m = kernel.compute_cross_K(self.support_X, test_X, argument)
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
            'kernel_centering_flag' : self.kernel_centering,
            'kernel_normalization_flag': self.kernel_normalization,
            'sv_masked_train_X': self.support_X,
            'sv_masked_train_Y': self.support_Y,
            'SMO_model': self.SMO_model}
        # --- SNO Save ---
        self.SMO_model.save_model(filepath)  #> Save the SMO model separately
        np.save(filepath + '_MKL_layer', model_data)
        # --- Kernel Save ---
        for i, kernel in enumerate(self.kernels_instances_matrix):
            kernel.save_model(filepath + f'_{i}_kernel')  #> Save each kernel separately


    def load_model(self, filepath):
        # --- Loading Model ---
        model_data = np.load(filepath + '_MKL_layer.npy', allow_pickle=True).item()
        self.kernels_arguments = model_data['kernels_arguments']
        self.selected_kernels = model_data['selected_kernels']
        self.kernel_centering = model_data['kernel_centering_flag']
        self.kernel_normalization = model_data['kernel_normalization_flag']
        self.support_X = model_data['sv_masked_train_X']
        self.support_Y = model_data['sv_masked_train_Y']
        # --- Loading SMO ---
        self.SMO_model = SMO()
        self.SMO_model.load_model(filepath)  #> Load the SMO model separately
        # --- Loading Kernels ---
        self.select_kernels(self.selected_kernels, #> Reinitialize kernels_instances_matrix based on selected_kernels
                            self.kernel_centering, 
                            self.kernel_normalization)  
        self.beta_array = model_data['beta_array']  #> Load beta_array from the saved model data
        # --- Loading Kernel Saved Data ---
        for kernel,i in zip(self.kernels_instances_matrix, range(len(self.kernels_instances_matrix))):
            kernel.load_model(filepath + f'_{i}_kernel.npy')  #> Load each kernel separately