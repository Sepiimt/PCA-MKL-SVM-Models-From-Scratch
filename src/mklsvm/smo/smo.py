import numpy as np
import warnings
from ..utils import timer


class SMO:
    def __init__(self):
        # --- Optimization Related ---
        self.b = None
        self.alpha_vector = None
        # --- Masks ---
        self.non_bound_mask = None
        self.non_bound_indices = None
        # --- Data ---
        self.train_kernel_matrix = None
        self.train_Y = None
        self.support_vector = None
        self.sv_indices = None
        # --- Utils ---
        self.timer = timer()
        # --- Flags ---
        self._indices_dirty = True
        self.converged = False
 

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


    def eta_computer(self, kernel_matrix, i, j):
        # --- Return Raw Eta, Remove the tolerance guard ---
        return 2.0 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
    

    def alpha_j_optimizer_and_clipper(self, Y, kernel_matrix, alpha_vector, errors_vector, i, j, eta, L, H, b):
        old_alpha_value_j, old_alpha_value_i = alpha_vector[j], alpha_vector[i]
        K_ii, K_jj, K_ij = kernel_matrix[i, i], kernel_matrix[j, j], kernel_matrix[i, j]
        # --- Standard Optimization ---
        if eta < 0:
            new_alpha_value_j = np.clip(old_alpha_value_j - (Y[j] * (errors_vector[i] - errors_vector[j])) / eta, L, H)
        # --- Fallback Objective Evaluation (eta >= 0) ---
        else:
            #> Uses the b passed in for this pair-update, not self.b, so the value
            #> can never silently drift from the bias actually active this cycle.
            f1 = Y[i] * (errors_vector[i] + b) - old_alpha_value_i * K_ii - old_alpha_value_j * Y[i] * Y[j] * K_ij
            f2 = Y[j] * (errors_vector[j] + b) - old_alpha_value_j * K_jj - old_alpha_value_i * Y[i] * Y[j] * K_ij
            L1 = old_alpha_value_i + Y[i] * Y[j] * (old_alpha_value_j - L)
            H1 = old_alpha_value_i + Y[i] * Y[j] * (old_alpha_value_j - H)
            # Objective functions at bounds L and H
            L_obj = L1 * f1 + L * f2 + 0.5 * (L1**2 * K_ii + L**2 * K_jj) + Y[i] * Y[j] * L * L1 * K_ij
            H_obj = H1 * f1 + H * f2 + 0.5 * (H1**2 * K_ii + H**2 * K_jj) + Y[i] * Y[j] * H * H1 * K_ij
            if L_obj < H_obj - 1e-4:
                new_alpha_value_j = L
            elif L_obj > H_obj + 1e-4:
                new_alpha_value_j = H
            else:
                new_alpha_value_j = old_alpha_value_j
        # --- Return ---
        return old_alpha_value_j, new_alpha_value_j
    

    def alpha_i_updater(self, Y, alpha_vector, old_alpha_value_j, new_alpha_value_j, i, j, C):
        old_alpha_value_i = alpha_vector[i]
        # Directional update based on class alignment
        new_alpha_value_i = old_alpha_value_i + Y[i] * Y[j] * (old_alpha_value_j - new_alpha_value_j)
        # --- Clipping New Alpha ---
        new_alpha_value_i = np.clip(new_alpha_value_i, 0.0, C) #> Clipping only to guard against floating-point drift.
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
            non_bound_mask, non_bound_indices = self.non_bound_mask_and_indices_creator(alpha_vector, tolerance, C)
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


    def non_bound_mask_and_indices_creator(self, alpha_vector, tolerance, C, only_calculate=False):
        # --- Compute Indices On-Demand for Outer Loops ---
        if self.non_bound_mask is None:
            self.non_bound_mask = (alpha_vector > tolerance) & (alpha_vector < C - tolerance)
            self._indices_dirty = True
        # --- Recompute Indices Only When The Mask Actually Changed ---
        #> Avoids an O(N) np.where() scan on every single call — this creator is
        #> invoked once per candidate `i` per pass, but the mask only changes
        #> when non_bound_mask_and_indices_updater() flips i or j.
        if self._indices_dirty:
            self.non_bound_indices = np.where(self.non_bound_mask)[0]
            self._indices_dirty = False
        # --- Return ---
        if not only_calculate:
            return self.non_bound_mask, self.non_bound_indices


    def non_bound_mask_and_indices_updater(self, C, tolerance, i, j):
        # --- Updating Values ---
        self.non_bound_mask[i] = (self.alpha_vector[i] > tolerance) & (self.alpha_vector[i] < C - tolerance)
        self.non_bound_mask[j] = (self.alpha_vector[j] > tolerance) & (self.alpha_vector[j] < C - tolerance)
        # --- Marking Indices Stale So The Next Creator Call Recomputes Them ---
        self._indices_dirty = True


    def j_vector_creator(self, alpha_vector, errors_vector, i, C, tolerance):
        # --- Calculating j Step Variances ---
        delta = np.abs(errors_vector - errors_vector[i])
        delta[i] = -np.inf
        # --- Yielding Primary Choice ---
        #> Maximum step size multiplier (Platt Heuristic)
        best_j = np.argmax(delta)
        yield best_j
        # --- Yielding Second Choice (Non-bound Values) ---
        #> Non-bound values (0 < alpha < C)
        non_bound_mask, non_bound_indices = self.non_bound_mask_and_indices_creator(alpha_vector, tolerance, C)
        non_bound_values = np.where(non_bound_mask)[0]
        #> Excluding pre-yielded or used values
        non_bound_values = non_bound_values[(non_bound_values != i) & (non_bound_values != best_j)]
        #> Randomizing scan order (Platt's heuristic) so the same subset
        #> is never starved by always being scanned in the same fixed order
        np.random.shuffle(non_bound_values)
        for value in non_bound_values:
            yield value
        # --- Yielding Third Choice (All left) ---
        #> Bound vectors (alpha == 0 or alpha == C)
        bound_values = np.where(~non_bound_mask)[0]
        #> Excluding pre-yielded or used values
        bound_values = bound_values[(bound_values != i) & (bound_values != best_j)]
        np.random.shuffle(bound_values)
        for value in bound_values:
            yield value


    def try_i_j_pair(self, Y, alpha_vector, errors_vector, kernel_matrix, tolerance, C, b, i, j):
        # --- Computing alpha Bounds (L,H) ----
        alpha_low_bound, alpha_high_bound = self.alpha_bounds_computer(alpha_vector, Y, i, j, C)
        if alpha_low_bound is None:
            return False
        # --- Computing Eta Value ---
        #> Note: eta_computer always returns a float (no tolerance guard by design,
        #> see its docstring), so no None-check is needed here.
        eta = self.eta_computer(kernel_matrix, i, j)
        # --- Clipping & Optimizing j Alpha ---
        old_alpha_value_j, new_alpha_value_j = self.alpha_j_optimizer_and_clipper(Y, kernel_matrix, alpha_vector, errors_vector, 
                                                                                i, j, eta, alpha_low_bound, alpha_high_bound, b)
        # --- Tiny Improvements Stop Gueard ---
        skip_i_j_pair = self.alpha_convergence_checker(old_alpha_value_j, new_alpha_value_j, tolerance)
        if skip_i_j_pair:
            return False
        # --- Updating i Alpha ---
        old_alpha_value_i, new_alpha_value_i = self.alpha_i_updater(Y, alpha_vector, old_alpha_value_j, new_alpha_value_j, i, j, C)
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
        # --- Updating Values ---
        self._values_updater(new_alpha_value_i, new_alpha_value_j, new_b, i, j)
        # --- Updating None-Bound Mask Vector ---
        self.non_bound_mask_and_indices_updater(C, tolerance, i, j)
        # --- Breaking The Loop After a Successful Cycle---
        return True
    

    def alpha_convergence_checker(self, old_alpha_value_j, new_alpha_value_j, tolerance):
        # --- Checking for Amount of Change in Alpha ---
        if abs(old_alpha_value_j-new_alpha_value_j) < tolerance:
            # --- Return ---
            return True
        # --- Return ---
        return False


    def fit(self, kernel_matrix, Y, C=1, max_iter=1000, tolerance=1e-5, 
            initial_alpha=None, initial_b=None, detailed_info=False):
        # --- Resetting Values ---
        #> Note: this always clears non_bound_mask/indices and train data — those
        #> must be rebuilt fresh per call regardless of warm-starting, since the
        #> kernel_matrix/Y passed in this call may differ from the previous one.
        self._value_reset()
        # --- Starting timer ---
        if detailed_info:
            self.timer.start()
        # --- Values Initialization ---
        self._values_initializer(Y, tolerance, C, initial_alpha=initial_alpha, initial_b=initial_b)
        # --- Saving Training Data ---
        self._save_train_data(kernel_matrix, Y)
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
            # --- Resetting Pass-Level Update Counter ---
            #> Counts every successful pair update made during this pass, rather than
            #> stopping at the first one — one violator's update no longer causes the
            #> rest of violators_mask to be skipped for this pass.
            num_changed = 0
            # --- Iterating Over Different Is
            for i in violators_mask:
                # --- Iterating Over Different Js ---
                for j in self.j_vector_creator(self.alpha_vector, errors_vector, i, C, tolerance):
                    # --- Passing Arguemnts to Sub-function ---
                    successful_cycle = self.try_i_j_pair(Y, self.alpha_vector, errors_vector, kernel_matrix, tolerance, C, self.b, i, j)
                    if successful_cycle:
                        num_changed += 1
                        #> This i is settled for this pass; move on to the next i.
                        break
            # --- State Alternation Logic ---
            if examine_all:
                if num_changed == 0:
                    #> We just scanned the entire dataset but could not make 
                    #> a single successful alpha update. Globally converged!
                    self.converged = True
                    break
                else:
                    #> We made successful updates; focus on non-bounds in the next pass
                    examine_all = False
            elif num_changed == 0:
                #> We scanned non-bounds but made zero updates; force a full pass
                examine_all = True
        # --- Stopping timer ---
        if detailed_info:
            self.timer.stop()
            sv_mask = self.alpha_vector > tolerance
            print(f"SMO.fit() Elapsed Time Is {self.timer.elapsed_time()} On {_+1} Iteration With {sum(sv_mask)} Support Vectors.")
            print(f"SMO.fit() New b: {self.b}")
        # --- Non-Convergence Notice ---
        #> Fires only if max_iter was exhausted without the KKT break condition
        #> ever being satisfied; self.converged stays False in that case.
        if not self.converged and detailed_info:
            warnings.warn(
                f"SMO.fit() did not converge within max_iter={max_iter} iterations "
                f"(tolerance={tolerance}). Results may be suboptimal; consider raising "
                f"max_iter or loosening tolerance.")
            
    def _value_reset(self):
        # --- Resetting Values ---
        self.b = None
        self.alpha_vector = None
        self.non_bound_mask = None
        self.non_bound_indices = None
        self._indices_dirty = True
        self.train_kernel_matrix = None
        self.train_Y = None
        self.converged = False
        
    def _save_train_data(self, kernel_matrix, Y):
        self.train_kernel_matrix = kernel_matrix
        self.train_Y = Y

    def _values_updater(self, new_alpha_value_i, new_alpha_value_j, new_b, i, j):
        # --- Updating Values ---
        self.alpha_vector[i] = new_alpha_value_i
        self.alpha_vector[j] = new_alpha_value_j
        self.b = new_b

    def _values_initializer(self, Y, tolerance, C, initial_alpha=None, initial_b=None):
        # --- Setting Values ---
        #> Warm-starting: reuse a previous solution's alpha/b instead of zeros, so SMO
        #> starts near the optimum. Meant for cases like MKL's outer beta-optimization
        #> loop, where the combined kernel only shifts slightly between successive
        #> fit() calls and a from-scratch restart wastes most of the passes redoing
        #> work already done last time.
        if initial_alpha is not None:
            initial_alpha = np.asarray(initial_alpha, dtype=float)
            if initial_alpha.shape != (len(Y),):
                raise ValueError(
                    f"initial_alpha has shape {initial_alpha.shape}, expected ({len(Y)},) to match Y.")
            self.alpha_vector = initial_alpha.copy()
            self.b = float(initial_b) if initial_b is not None else 0.0
        else:
            self.alpha_vector = np.zeros(len(Y))
            self.b = 0.0
        self.non_bound_mask_and_indices_creator(self.alpha_vector, tolerance, C, only_calculate=True)
    

    def extract_support_vectors(self, Y, tolerance):
        # --- Identify Non-Zero Alphas ---
        sv_mask = self.alpha_vector > tolerance
        # --- Save Only Support Vector Data ---
        self.sv_indices = np.where(sv_mask)[0] #> MKL needs these to filter the test kernel
        self.support_vector = self.alpha_vector[sv_mask]
        self.support_Y = Y[sv_mask]
        # --- Purge Full Kernel Matrix to Free Memory ---
        self.train_kernel_matrix = None


    def predict(self, K_test):
            # --- Initializing Values ---
            coefficients = self.support_vector * self.support_Y
            # --- Mathematical Computation ---
            raw_decision_scores = np.dot(K_test, coefficients) + self.b
            # --- Return ---
            return raw_decision_scores


    def save_model(self, filepath):
        model_dict = {
            'b': self.b,
            'support_vector': self.support_vector,
            'sv_masked_train_Y': self.support_Y,
            'sv_indices': self.sv_indices}
        np.save(filepath + '_SMO_layer', model_dict)


    def load_model(self, filepath):
        # --- Loading Model ---
        loaded_data = np.load(filepath + '_SMO_layer.npy', allow_pickle=True).item()
        self.b = loaded_data['b']
        self.support_vector = loaded_data['support_vector']
        self.support_Y = loaded_data['sv_masked_train_Y']
        self.sv_indices = loaded_data['sv_indices']