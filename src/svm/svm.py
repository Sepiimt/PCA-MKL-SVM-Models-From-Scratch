import numpy as np
from mkl import MKL
from utils import Timer
import os

class SVM:
    # --- Documentation ---
    """
    --- SVM Class ---
    Multi Kernel Learning SVM algorithm implemented from scratch by "Sepanta Metanat"

    First edit: "2026/06/30"
    Last edit: "2026/06/30"
    """

    def __init__(self):
        self.selected_kernels = None
        self.kernels_arguments = None
        self.MKL_instance = MKL()

    def select_kernels(self, selected_kernels):
        # --- Documentation ---
        """
        Available Kernels: 1.Linear, 2.Polynomial, 3.RBF, 4.Sigmoid
        Simply parse an array with the length of available kernels and set "True" for your desirable kernels you wish to be included in MKL.

        Example: [False, True, True, True]
        """
        # --- Checking for Input Validity ---
        if len(selected_kernels) != 4 or not all(isinstance(i, bool) for i in selected_kernels):
            raise ValueError("Error: Please provide a boolean array with the length of 4 for selecting kernels.")
        # --- Setting Value ---
        self.selected_kernels = selected_kernels
        self.MKL_instance.select_kernels(self.selected_kernels)
    
    def sub_set_kernels_arguments_values_granter(self):
        # --- Priting Info ---
        print('Enter "default" for setting default value for each input')
        # --- if "Polynomial" Selected ---
        if self.selected_kernels[1]:
            print('For "Polynomial" Kernel:')
            poly_c = float(input('Enter "c" Value: '))
            poly_d = float(input('Enter "d" Value: '))
        # --- if "RBF" Selected ---
        if self.selected_kernels[2]:
            print('For "RBF" Kernel:')
            rbf_gamma = float(input('Enter "gamma" Value: '))
        # --- if "Sigmoid" Selected ---
        if self.selected_kernels[3]:
            print('For "Sigmoid" Kernel:')
            sigmoid_gamma = float(input('Enter "gamma" Value: '))
            sigmoid_c = float(input('Enter "c" Value: '))
        # --- Packing into a Single NumPy Vector ---
        kernel_params = np.array([poly_c, poly_d, rbf_gamma, sigmoid_gamma, sigmoid_c], dtype=np.float64)
        # --- Return ---
        return kernel_params

    def sub_set_kernels_arguments_default_values(self, X):
        # --- Setting Default Values ---
        n_features = X.shape[1]
        data_variance = X.var()
        gamma_scale = 1.0 / (n_features * data_variance) if data_variance != 0 else 1.0
        #> Preventing Division by zero if data has zero variance
        poly_c = 0.0  #> Default 'coef0' for Polynomial
        poly_d = 3.0  #> Default 'degree' for Polynomial
        rbf_gamma = gamma_scale
        sigmoid_gamma = gamma_scale
        sigmoid_c = 0.0  #> Default 'coef0' for Sigmoid
        # --- Packing into a Single NumPy Vector ---
        kernel_params = np.array([poly_c, poly_d, rbf_gamma, sigmoid_gamma, sigmoid_c], dtype=np.float64)
        # --- Return ---
        return kernel_params

    def set_kernels_arguments(self, X, default_values=True):
        # --- Documentation ---
        """
        Simply run the method and answer the questions, or set "default_values=True" to use default values.
        """
        # --- Checking for Selected Kernels ---
        if self.selected_kernels is None:
            raise ValueError("Error: Please select the kernels first by calling 'SVM.select_kernels()' method.")
        if not default_values:
            self.kernels_arguments = self.sub_set_kernels_arguments_values_granter()
        else:
            # --- Setting Default Values ---
            self.kernels_arguments = self.sub_set_kernels_arguments_default_values(X)      

    def sub_fit_input_validator(self, X, Y):
        Y_unique_values = np.unique(Y)
        if len(Y_unique_values) != 2:
            raise ValueError("Error: Y array has additional values other than +1/-1!")
        if np.sum(Y_unique_values) != 0:
            raise ValueError("Error: Y array's values have not turned into +1/-1!")
        if len(X) != len(Y):
            raise ValueError("Error: Length of Y & X array does not match!")

    def sub_fit_kernels_arguments_validator(self):
        if self.kernels_arguments is None or self.selected_kernels is None:
            raise ValueError("Error: Please set the kernel arguments first by calling 'SVM.set_kernels_arguments()' method.")

    def fit(self, X, Y,
            beta_learning_rate = 0.1, max_beta_optimization_iter = 500, beta_optimization_tolerance = 1e-3,
            smo_C = 1, max_SMO_iter = 1000, SMO_tolerance = 1e-3,
            general_timer=False, detailed_timer=False):
        # --- Documentation ---
        """
        #> Parameters Documentation:
        1. X: training x_array
        2. Y: training y_array
        3. beta_learning_rate: Learning rate of "Beta" for MKL
        4. max_beta_optimization_iter: Maximum iteration limit for "Beta" Optimization in MKL
        5. SMO_tolerance: Minimum required changes in "Beta" in MKL
        6. smo_C: "C" parameter for SMO
        7. max_SMO_iter: Iteration limit for optimizing "Alpha Vector" in SMO
        8. SMO_tolerance: Minimum required changes in "Alpha Vector" in SMO
        9. general_timer: If True, will print the elapsed time of SVM.fit() function
        10. detailed_timer: If True, will print the elapsed time of all underlaying functions of SVM.fit() function

        #> Attention: Be sure to call "SVM.select_kernels()" and "SVM.kernels_arguments()" before calling "SVM.fit()" function.
        """
        # --- Validating Kernel Arguments ---
        self.sub_fit_kernels_arguments_validator()
        # --- Validating the Input ---
        self.sub_fit_input_validator(X, Y)
        # --- Setting Timer ---
        if general_timer:
            self.MKL_instance.Timer.start()
        # --- Parsing to MKL ---
        self.MKL_instance.fit(X, Y, 
                              smo_C, max_SMO_iter, SMO_tolerance, 
                              beta_learning_rate, self.kernels_arguments, 
                              max_beta_optimization_iter = max_beta_optimization_iter, 
                              beta_optimization_tolerance = beta_optimization_tolerance,
                              detailed_timer = detailed_timer)
        # --- Stopping Timer ---
        if general_timer:
            self.MKL_instance.Timer.stop()
            print(f"SVM.fit() Elapsed Time: {self.MKL_instance.Timer.elapsed_time()}")
        
    def predict(self, test_X, strict_binary_result=False):
        # --- Documentation ---
        """
        #> Available Arguments:
        Positional Arguments: X
        Optional Arguments: strict_binary_result

        Output: predicted_y 
        """
        # --- Computing Results ---
        raw_scores = self.MKL_instance.predict(test_X)
        if strict_binary_result:
            # --- Return ---
            return np.sign(raw_scores) #> Convert continuous distance scores into strict binary classifications
        # --- Return ---
        return raw_scores
    
    def save_model(self, filepath):
        filepath = os.path.join(filepath, "model_data") 
        model_data = {
            'selected_kernels': self.selected_kernels,
            'kernels_arguments': self.kernels_arguments}
        np.save(filepath + '_SVM_layer', model_data)
        # --- Saving Model ---
        self.MKL_instance.save_model(filepath)

    def load_model(self, filepath):
        filepath = os.path.join(filepath, "model_data")
        # --- Loading Model ---
        model_data = np.load(filepath + '_SVM_layer.npy', allow_pickle=True).item()
        self.selected_kernels = model_data['selected_kernels']
        self.kernels_arguments = model_data['kernels_arguments']
        self.MKL_instance.load_model(filepath)
