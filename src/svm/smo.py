import numpy as np
from utils import Timer

class SMO:
    def __init__(self):
        self.b = None
        self.alpha_vector = None
        self.Timer = Timer()
        self.non_bound_mask = None
        self.non_bound_indices = None
        self.train_kernel_matrix = None
        self.train_Y = None
 
    def alpha_bounds_computer(self, alpha_vector, Y, i, j, C):
        # --- Determining Values ---
        alpha_value_i, alpha_value_j = alpha_vector[i], alpha_vector[j]
        Yi, Yj = Y[i], Y[j]
        # --- Calculating High & Low Cap for Alpha ---
        if Yi != Yj:
            alpha_low_bound = max(0.0, alpha_value_j - alpha_value_i)
            alpha_high_bound = min(C, C + alpha_value_j - alpha_value_i)
        else:
            alpha_low_bound = max(0.0, alpha_value_i + alpha_value_j - C)
            alpha_high_bound = min(C, alpha_value_i + alpha_value_j)
        # --- Unacceptable Value Guard ---
        if alpha_low_bound == alpha_high_bound:
            return None, None
        # --- Return ---
        return alpha_low_bound, alpha_high_bound

    def errors_vector_computer(self, Y, alpha_vector, kernel_matrix, b):
        # Computes f(x) - y for every single sample in one operation in the shape of (N,)
        #> Note: alphas will be changed after the whole loop, thus calculating error_matirx at start is optimized
        errors_vector = np.dot(alpha_vector * Y, kernel_matrix) + b - Y
        return errors_vector
    
    def error_vector_updater(self, Y, errors_vector, kernel_matrix,
                             old_alpha_value_i, new_alpha_value_i, 
                             old_alpha_value_j, new_alpha_value_j, 
                             b, new_b, i, j):
        # --- Calculating Differences (Deltas) ---
        delta_alpha_i = new_alpha_value_i - old_alpha_value_i
        delta_alpha_j = new_alpha_value_j - old_alpha_value_j
        delta_b = new_b - b
        # --- Calculating New Error Vector ---
        errors_vector += (delta_alpha_i * Y[i] * kernel_matrix[i]+ delta_alpha_j * Y[j] * kernel_matrix[j]+ delta_b)
        # --- return ---
        return errors_vector

    def eta_computer(self, kernel_matrix, i, j, tolerance):
        # If eta >= 0, the optimization step is invalid; skip this pair
        eta = 2.0 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
        # --- Unacceptable Value Guard ---
        if abs(eta) < tolerance :
            return None
        # --- Return ---
        return eta
    
    def alpha_j_optimizer_and_clipper(self, Y, alpha_vector, errors_vector, i, j, eta, L, H):
        # Save the old value for delta tracking later
        old_alpha_value_j = alpha_vector[j]
        # Analytical update step + constraint enforcement
        new_alpha_value_j = np.clip(old_alpha_value_j - (Y[j] * (errors_vector[i] - errors_vector[j])) / eta, L, H)
        # --- Return---
        return old_alpha_value_j, new_alpha_value_j
    
    def alpha_i_updater(self, Y, alpha_vector, old_alpha_value_j, new_alpha_value_j, i, j):
        old_alpha_value_i = alpha_vector[i]
        # Directional update based on class alignment
        new_alpha_value_i = old_alpha_value_i + Y[i] * Y[j] * (old_alpha_value_j - new_alpha_value_j)
        # --- Return ---
        return old_alpha_value_i, new_alpha_value_i

    def bias_updater(self, Y, kernel_matrix, errors_vector, 
                     old_alpha_value_i, new_alpha_value_i, 
                     old_alpha_value_j, new_alpha_value_j,
                     i, j, b, C):
        # --- Calculating alpha's difference (Delta) ---
        delta_i = new_alpha_value_i - old_alpha_value_i
        delta_j = new_alpha_value_j - old_alpha_value_j
        # --- Calculating Potential Biases (Thresholds) ---
        b1 = b - errors_vector[i] - Y[i] * delta_i * kernel_matrix[i, i] - Y[j] * delta_j * kernel_matrix[i, j]
        b2 = b - errors_vector[j] - Y[i] * delta_i * kernel_matrix[i, j] - Y[j] * delta_j * kernel_matrix[j, j]
        # --- Updating Global Bias ---
        if 0.0 < new_alpha_value_i < C:
            new_b = b1
        elif 0.0 < new_alpha_value_j < C:
            new_b = b2
        else:
            new_b = (b1 + b2) / 2.0
        # --- Return ---
        return new_b

    def KKT_condition_violators_computer(self, Y, alpha_vector, errors_vector, C, tolerance):
        # --- Calculating Margin ---
        margin = (errors_vector + Y) * Y #> Criteria
        # --- Violation Mask ---
        #> Violations Mask for Where alpha Should be Larger but Margin is Too Small
        violation_low = (alpha_vector < C) & (margin < 1.0 - tolerance)
        #> Violations Mask for Where alpha Should be Zero but Margin is Too Large
        violation_high = (alpha_vector > 0) & (margin > 1.0 + tolerance)
        #> Combining boolean masks
        all_violations = violation_low | violation_high
        # --- Return ---
        #> Extracting the Physical Indices Where True Occurs
        return np.where(all_violations)[0]

    def LIBSVM_KKT_computer(self, Y, alpha_vector, errors_vector, C, tolerance, examine_all):
        # --- LIBSVM Strategy Selector ---
        if not examine_all:
            #> Sift out only the non-bound support vectors (0 < alpha < C)
            non_bound_mask, non_bound_indices = self.sub_fit_non_bound_mask_and_indices_creator(alpha_vector, tolerance, C)
            # --- Checking For Empty None_Bound Mask ---
            if len(self.non_bound_indices) == 0:
                return np.array([], dtype=int)
            # --- Extracting None-Bound Violators ---
            #> Find KKT violations inside this specific subset (non-bound values)
            sub_violators = self.KKT_condition_violators_computer(
                Y[non_bound_indices], alpha_vector[non_bound_indices], errors_vector[non_bound_indices], C, tolerance)
            # --- Return ---
            #> Maping back indices to the global dataset matrix space
            return non_bound_indices[sub_violators]
        # --- Normal KKT Computation ---
        else:
            # --- Return ---
            #> Standard execution: check entire dataset
            return self.KKT_condition_violators_computer(Y, alpha_vector, errors_vector, C, tolerance)

    def sub_fit_values_initializer(self, Y, tolerance, C):
        # --- Setting Values ---
        self.alpha_vector = np.zeros(len(Y))
        self.b = 0.0
        self.sub_fit_non_bound_mask_and_indices_creator(self.alpha_vector, tolerance, C, only_calculate=True)
    
    def sub_fit_values_updater(self, new_alpha_value_i, new_alpha_value_j, new_b, i, j):
        # --- Updating Values ---
        self.alpha_vector[i] = new_alpha_value_i
        self.alpha_vector[j] = new_alpha_value_j
        self.b = new_b

    def sub_fit_non_bound_mask_and_indices_creator(self, alpha_vector, tolerance, C, only_calculate=False):
        # --- Creating None-Bound Mask Vector ---
        if self.non_bound_mask is None or self.non_bound_indices is None:
                self.non_bound_mask = (alpha_vector > tolerance) & (alpha_vector < C - tolerance)
                self.non_bound_indices = np.where(self.non_bound_mask)[0]
                if not only_calculate:
                    return self.non_bound_mask, self.non_bound_indices
        # --- Returning if Available ---
        else:
            return self.non_bound_mask, self.non_bound_indices

    def sub_fit_non_bound_mask_and_indices_updater(self, C, tolerance, i, j):
        # --- Updating Values ---
        self.non_bound_mask[i] = (self.alpha_vector[i] > tolerance) & (self.alpha_vector[i] < C - tolerance)
        self.non_bound_mask[j] = (self.alpha_vector[j] > tolerance) & (self.alpha_vector[j] < C - tolerance)
        self.non_bound_indices = np.where(self.non_bound_mask)[0]

    def sub_fit_j_vector_creator(self, alpha_vector, errors_vector, i, C, tolerance):
        # --- Calculating j Step Variances ---
        delta = np.abs(errors_vector - errors_vector[i])
        delta[i] = -np.inf
        # --- Yielding Primary Choice ---
        #> Maximum step size multiplier (Platt Heuristic)
        best_j = np.argmax(delta)
        yield best_j
        # --- Yielding Second Choice (Non-bound Values) ---
        #> Non-bound values (0 < alpha < C)
        non_bound_mask, non_bound_indices = self.sub_fit_non_bound_mask_and_indices_creator(alpha_vector, tolerance, C)
        non_bound_values = np.where(non_bound_mask)[0]
        #> Excluding pre-yielded or used values
        non_bound_values = non_bound_values[(non_bound_values != i) & (non_bound_values != best_j)]
        for value in non_bound_values:
            yield value
        # --- Yielding Third Choice (All left) ---
        #> Bound vectors (alpha == 0 or alpha == C)
        bound_values = np.where(~non_bound_mask)[0]
        #> Excluding pre-yielded or used values
        bound_values = bound_values[(bound_values != i) & (bound_values != best_j)]
        for value in bound_values:
            yield value

    def alpha_change_is_too_small(self, old_alpha_value_j, new_alpha_value_j, tolerance):
        # --- Checking for Amount of Change in Alpha ---
        if abs(old_alpha_value_j-new_alpha_value_j) < tolerance:
            # --- Return ---
            return True
        # --- Return ---
        return False
    
    def sub_fit_save_train_data(self, kernel_matrix, Y):
        self.train_kernel_matrix = kernel_matrix.copy()
        self.train_Y = Y.copy()

    def sub_fit_try_i_j_pair(self, Y, alpha_vector, errors_vector, kernel_matrix, tolerance, C, b, i, j):
        # --- Computing alpha Bounds (L,H) ----
        alpha_low_bound, alpha_high_bound = self.alpha_bounds_computer(alpha_vector, Y, i, j, C)
        if alpha_low_bound is None:
            return False
        # --- Computing Eta Value ---
        eta = self.eta_computer(kernel_matrix, i, j, tolerance)
        if eta is None:
            return False
        # --- Clipping & Optimizing j Alpha ---
        old_alpha_value_j, new_alpha_value_j = self.alpha_j_optimizer_and_clipper(Y, alpha_vector, errors_vector, 
                                                                                i, j, eta, alpha_low_bound, alpha_high_bound)
        # --- Tiny Improvements Stop Gueard ---
        skip_i_j_pair = self.alpha_change_is_too_small(old_alpha_value_j, new_alpha_value_j, tolerance)
        if skip_i_j_pair:
            return False
        # --- Updating i Alpha ---
        old_alpha_value_i, new_alpha_value_i = self.alpha_i_updater(Y, alpha_vector, old_alpha_value_j, new_alpha_value_j, i, j)
        # --- Updating Bisa ---
        new_b = self.bias_updater(Y, kernel_matrix, errors_vector, 
                                old_alpha_value_i, new_alpha_value_i, 
                                old_alpha_value_j, new_alpha_value_j,
                                i, j, b, C)
        # --- Error Vector Update ---
        errors_vector = self.error_vector_updater(Y, errors_vector, kernel_matrix,
                                                old_alpha_value_i, new_alpha_value_i, 
                                                old_alpha_value_j, new_alpha_value_j, 
                                                b, new_b, i, j)
        # --- Updating None-Bound Mask Vector ---
        self.sub_fit_non_bound_mask_and_indices_updater(C, tolerance, i, j)
        # --- Updating Values ---
        self.sub_fit_values_updater(new_alpha_value_i, new_alpha_value_j, new_b, i, j)
        # --- Breaking The Loop After a Successful Cycle---
        return True

    def sub_fit_SMO_value_reset(self):
        # --- Resetting Values ---
        self.b = None
        self.alpha_vector = None
        self.non_bound_mask = None
        self.non_bound_indices = None
        self.train_kernel_matrix = None
        self.train_Y = None

    def fit(self, kernel_matrix, Y, C=1, max_iter=1000, tolerance=1e-3, detailed_timer=False):
        # --- Resetting Values ---
        self.sub_fit_SMO_value_reset()
        # --- Starting Timer ---
        if detailed_timer:
            self.Timer.start()
        # --- Values Initialization ---
        self.sub_fit_values_initializer(Y, tolerance, C)
        # --- Saving Training Data ---
        self.sub_fit_save_train_data(kernel_matrix, Y)
        # --- Initializing Error Vector ---
        errors_vector = self.errors_vector_computer(Y, self.alpha_vector, kernel_matrix, self.b)
        # --- LIBSVM Working-Set State Flag ---
        examine_all = True
        # --- Training Loop ---
        for _ in range(max_iter):
            # --- Creating Violators Mask ---
            violators_mask = self.LIBSVM_KKT_computer(Y, self.alpha_vector, errors_vector, C, tolerance, examine_all)
            # --- Break Criteria ---
            if len(violators_mask) == 0:
                if examine_all:
                    #> Globally converged across the whole dataset
                    break
                else:
                    #> Non-bound space completed; force a full pass to check bounds
                    examine_all = True
                    continue
            # --- Resetting Loop Stop Guard Value ---
            successful_cycle = False
            # --- Iterating Over Different Is
            for i in violators_mask:
                # --- Successful Cycle Stop Gueard ---
                if successful_cycle:
                    break
                # --- Iterating Over Different Js ---
                for j in self.sub_fit_j_vector_creator(self.alpha_vector, errors_vector, i, C, tolerance):
                    # --- Passing Arguemnts to Sub-function ---
                    successful_cycle = self.sub_fit_try_i_j_pair(Y, self.alpha_vector, errors_vector, kernel_matrix, tolerance, C, self.b, i, j)
                    if successful_cycle:
                        break
            # --- State Alternation Logic ---
            if examine_all:
                #> We Finished a Full Dataset Check, Switching Back to Focusing On non-bounds
                examine_all = False
            elif not successful_cycle:
                #> We Passed On non-bounds Which Yielded to Zero Optimizations, Switching to Full Check
                examine_all = True
        # --- Stopping Timer ---
        if detailed_timer:
            self.Timer.stop()
            print(f"SMO.fit() Elapsed Time: {self.Timer.elapsed_time()}")

    def predict(self, K_test):
            # --- Initializing Values ---
            #> K_test shape: (n_train, n_test)
            #> alpha_vector * Y_train shape: (n_train,)
            coefficients = self.alpha_vector * self.train_Y
            # --- Mathematical Computation ---
            #> Vectorized matrix product yields raw scores vector of shape: (n_test,)
            raw_decision_scores = np.dot(coefficients, K_test) + self.b
            # --- Return ---
            return raw_decision_scores
    
    def save_model(self, filepath):
        # --- Saving Model ---
        model_dict = {
            'b': self.b,
            'alpha_vector': self.alpha_vector,
            'non_bound_mask': self.non_bound_mask,
            'non_bound_indices': self.non_bound_indices,
            'train_kernel_matrix': self.train_kernel_matrix,
            'train_Y': self.train_Y}
        np.save(filepath + '_SMO_layer', model_dict)

    def load_model(self, filepath):
        # --- Loading Model ---
        loaded_data = np.load(filepath + '_SMO_layer.npy', allow_pickle=True).item()
        self.b = loaded_data['b']
        self.alpha_vector = loaded_data['alpha_vector']
        self.non_bound_mask = loaded_data['non_bound_mask']
        self.non_bound_indices = loaded_data['non_bound_indices']
        self.train_kernel_matrix = loaded_data['train_kernel_matrix']
        self.train_Y = loaded_data['train_Y']