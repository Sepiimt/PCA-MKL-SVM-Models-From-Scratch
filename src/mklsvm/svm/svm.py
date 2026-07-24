import numpy as np
import os
from ..mkl import MKL
from ..utils import timer


class SVM:
    # --- Documentation ---
    """
    --- SVM Class ---
    Multi Kernel Learning SVM algorithm implemented from scratch by "Sepanta Metanat"

    First edit: "2026/06/30"
    Last edit: "2026/07/24"
    """

    def __init__(self):
        self.selected_kernels = None
        self.default_selected_kernels = [True,True,True,True,True,False] #> Sigmoid is disabled by default.
        self.kernels_arguments = None
        self.MKL_instance = MKL()
        # --- Available Beta Optimization Methods (BOM) ---
        self.available_bom = ["SimpleMKL", "L2-MKL", "Entropic Descent", "Hybrid L1/L2", "KTA", "Frank-Wolfe", "FISTA"]
        # --- Default Beta Optimization Method (BOM ---)
        self.bom = "Entropic Descent"


    def select_kernels(self, selected_kernels = None, centering = False, normalization = True):
        # --- Documentation ---
        """
        #> Attention: Do Not call or give input if you wish to proceed with default selected kernels.

        Available Kernels: 1.Linear, 2.Polynomial, 3.RBF, 4.Laplacian, 5.Rational Quadratic, 6.Sigmoid
        Simply parse an array with the length of available kernels and set "True" for your desirable kernels you wish to be included in MKL.
        You can Enable/Disable kernel matrices "centering" and "normalization" by optional flag.
        
        Default: [True, True, True, True, True, False]
        """
        # --- Checking for Input Validity ---
        if selected_kernels is not None:
            if len(selected_kernels) != 6 or not all(isinstance(i, bool) for i in selected_kernels):
                raise ValueError("Error: Please provide a boolean array with the length of 6 for selecting kernels.")
        else:
            selected_kernels = self.default_selected_kernels
        # --- Setting Value ---
        self.selected_kernels = selected_kernels
        self.MKL_instance.select_kernels(self.selected_kernels, centering = centering, normalization = normalization)
    

    def set_kernels_arguments(self, X, default_values = False):
        # --- Documentation ---
        """
        #> Attention: Do Not call or set "default_values=True" if you wish to proceed with default selected kernels.

        Simply run the method and answer the questions if you wish to modify the kernle's default values.
        """
        # --- Checking for Selected Kernels ---
        if self.selected_kernels is None:
            raise ValueError("Error: Please select the kernels first by calling 'SVM.select_kernels()' method.")
        if not default_values:
            self.kernels_arguments = self._arguments_values_granter()
        else:
            # --- Setting Default Values ---
            self.kernels_arguments = self._arguments_default_values(X)

    def guments_values_granter(self):
        # --- Priting Info ---
        print('Enter a value for each input')
        # --- if "Polynomial" Selected ---
        if self.selected_kernels[1]:
            print('For "Polynomial" Kernel:')
            poly_c = float(input('Enter "c" Value: '))
            poly_d = float(input('Enter "d" Value: '))
        # --- if "RBF" Selected ---
        if self.selected_kernels[2]:
            print('For "RBF" Kernel:')
            rbf_gamma = float(input('Enter "gamma" Value: '))
        # --- if Laplacian Kernel ---
        if self.selected_kernels[3]:
            print('For "Laplacian" Kernel:')
            lap_gamma = float(input('Enter "gamma" Value: '))
        # --- if Rational Quadratic Kernel ---
        if self.selected_kernels[4]:
            print('For "Rational Quadratic" Kernel:')
            rq_alpha = float(input('Enter "alpha" Value: '))
            rq_ls = float(input('Enter "length_scale" Value: '))
        # --- if "Sigmoid" Selected ---
        if self.selected_kernels[5]:
            print('For "Sigmoid" Kernel:')
            sigmoid_gamma = float(input('Enter "gamma" Value: '))
            sigmoid_c = float(input('Enter "c" Value: '))
        # --- Packing into a Single NumPy Vector ---
        kernel_params = np.array([poly_c, poly_d, rbf_gamma, lap_gamma, rq_alpha, rq_ls, sigmoid_gamma, sigmoid_c], dtype=np.float64)
        # --- Return ---
        return kernel_params

    def _arguments_default_values(self, X):
        # --- Setting Default Values ---
        n_features = X.shape[1]
        data_variance = X.var()
        data_mean_sq = np.mean(X**2)  #> E[x^2] = variance + mean^2; governs raw dot-product magnitude
        gamma_scale = 1.0 / (n_features * data_variance) if data_variance != 0 else 1.0
        sigmoid_gamma_scale = 1.0 / (n_features * data_mean_sq) if data_mean_sq != 0 else 1.0
        #> Preventing Division by zero if data has zero variance
        poly_c = 0.0  #> Default 'coef0' for Polynomial
        poly_d = 3.0  #> Default 'degree' for Polynomial
        rbf_gamma = gamma_scale
        sigmoid_gamma = sigmoid_gamma_scale
        sigmoid_c = 0.0  #> Default 'coef0' for Sigmoid
        lap_gamma = gamma_scale
        rq_alpha = 1.0
        rq_ls = np.sqrt(n_features * data_variance) if data_variance != 0 else 1.0
        #> length_scale must grow with dimensionality/variance for the same reason gamma
        #> must shrink — a fixed value collapses off-diagonal similarity as n_features grows
        # --- Packing into a Single NumPy Vector ---
        kernel_params = np.array([poly_c, poly_d, rbf_gamma, lap_gamma, rq_alpha, rq_ls, sigmoid_gamma, sigmoid_c], dtype=np.float64)
        # --- Return ---
        return kernel_params      


    def set_beta_optimizer(self, method = "Entropic Descent"):
        # --- Documentation ---
        """
        #> Attention: Do Not call this function if you wish to proceed with default ("Entropic Descent") beta optimization algorithm.

        Available Beta Optimization Methods:
        "SimpleMKL", "L2-MKL", "Entropic Descent", "Hybrid L1/L2", "KTA", "Frank-Wolfe", "FISTA"
        """
        # --- Checking for Selected BOM ---
        if method not in self.available_bom:
            raise ValueError("Error: Select a Valid Beta Optimization Method!")
        else:
            self.bom = method


    def fit(self, X, Y,
            max_bo_iter = 500, 
            bo_tolerance = 1e-3,
            smo_c = 1, 
            max_smo_iter = 2000, 
            smo_tolerance = 1e-5,
            km_centering = False, 
            km_normalization = True,
            general_timer = True, 
            detailed_info = False):
        # --- Documentation ---
        """
        #> Parameters Documentation:
        1. X: training x_array
        2. Y: training y_array
        3. max_bo_iter: Maximum iteration limit for "Beta" Optimization in MKL
        4. bo_tolerance: Minimum required changes in "Beta" in MKL
        5. smo_c: "C" parameter for SMO
        6. max_smo_iter: Iteration limit for optimizing "Alpha Vector" in SMO
        7. smo_tolerance: Minimum required changes in "Alpha Vector" in SMO
        8. km_centering: Boolian value for enabeling kernel matrices centering. Used only if "MKL.select_kernels()" hasn't been run.
        9. km_normalization: Boolian value for enabeling kernel matrices diagnol normalization. Used only if "MKL.select_kernels()" hasn't been run.
        10. general_timer: If True, will print the elapsed time of SVM.fit() function
        11. detailed_info: If True, will print the elapsed time of all underlaying functions of SVM.fit() function with their train details

        #> Attention: Be sure to call "SVM.select_kernels()", "SVM.kernels_arguments()" or "SVM.set_beta_optimizer()" before calling "SVM.fit()" for customizing the model.
        """
        # --- Validating Kernel Arguments ---
        self._kernels_info_validator(X, km_centering, km_normalization)
        # --- Validating the Input ---
        self._input_validator(X, Y)
        # --- Setting timer ---
        if general_timer and not detailed_info:
            self.MKL_instance.timer.start()
        # --- Parsing to MKL ---
        self.MKL_instance.fit(X, Y, 
                              smo_c,
                              max_smo_iter,
                              smo_tolerance, 
                              self.kernels_arguments,
                              beta_optimization_method = self.bom,
                              max_bo_iter = max_bo_iter, 
                              bo_tolerance = bo_tolerance,
                              detailed_info = detailed_info)
        # --- Stopping timer ---
        if general_timer and not detailed_info:
            self.MKL_instance.timer.stop()
            print(f"SVM.fit() Elapsed Time: {self.MKL_instance.timer.elapsed_time()}.")

    def _input_validator(self, X, Y):
        Y_unique_values = np.unique(Y).astype(int)
        if len(Y_unique_values) != 2:
            raise ValueError("Error: Y array has additional values other than +1/-1!")
        if np.sum(Y_unique_values) != 0:
            raise ValueError("Error: Y array's values have not turned into +1/-1!")
        if len(X) != len(Y):
            raise ValueError("Error: Length of Y & X array does not match!")

    def _kernels_info_validator(self, X, centering, normalization):
        # --- Selecting Default Kerlens ---
        if self.selected_kernels is None:
            self.select_kernels(self.default_selected_kernels, centering, normalization)
        # --- Setting Default Values ---
        if self.kernels_arguments is None:
            self.set_kernels_arguments(X, default_values=True)
            

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
