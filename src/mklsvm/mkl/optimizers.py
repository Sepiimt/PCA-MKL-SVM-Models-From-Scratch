import numpy as np


class BaseBetaOptimizer:
    """Abstract base class for beta optimization strategies."""
    def update(self, support_vector_weights, Y, kernels_instances_matrix, current_beta_array, **kwargs):
        raise NotImplementedError("Subclasses must implement the update method.")

    def gradient_computer(self, support_vector_weights, Y, kernels_instances_matrix):
        # --- Value Initialization ---
        weighted_Y = support_vector_weights * Y
        label_outer_product = np.outer(weighted_Y, weighted_Y)
        # --- Vectorized Gradient Calculation ---
        gradient = -0.5 * np.array([np.sum(label_outer_product * km.K) for km in kernels_instances_matrix])
        # --- Return ---
        return gradient

    def simplex_projector(self,gradient_step, num_kernels):
        # --- Simplex Projection ---
        sorted_step = np.sort(gradient_step)[::-1]
        cumulative_sum = np.cumsum(sorted_step)
        active_idx = np.nonzero(sorted_step * np.arange(1, num_kernels + 1) > (cumulative_sum - 1))[0][-1]
        projection_threshold = (cumulative_sum[active_idx] - 1) / (active_idx + 1.0)
        # --- Return ---
        return projection_threshold



class ReducedGradientOptimizer(BaseBetaOptimizer):
    """
    AKA: SimpleMKL, Active Set Method, Steepest Descent on the Simplex.
    Goal: Standard SimpleMKL update for L1-norm (Sparsity).
    Technique: Forces beta to sum to 1, pushing unhelpful kernels to exactly 0.
    """
    def __init__(self, step_size=0.1):
        self.step_size = step_size  

    def update(self, support_vector_weights, Y, kernels_instances_matrix, current_beta_array, **kwargs):
        # --- Value Initialization ---
        gradient = self.gradient_computer(support_vector_weights, Y, kernels_instances_matrix)
        # --- Locating Optimal Reference Index ---
        active_mask = current_beta_array > 1e-8
        if not np.any(active_mask):
            reference_idx = np.argmin(gradient)
        else:
            masked_gradient = np.where(active_mask, gradient, np.inf)
            reference_idx = np.argmin(masked_gradient)
        reference_gradient = gradient[reference_idx]
        # --- Vectorized Descent Direction Calculation ---
        reduced_gradient = gradient - reference_gradient
        descent_direction = -reduced_gradient
        # --- KKT Boundary Lock ---
        boundary_mask = (current_beta_array <= 1e-8) & (reduced_gradient > 0)
        descent_direction[boundary_mask] = 0.0
        descent_direction[reference_idx] = 0.0
        descent_direction[reference_idx] = -np.sum(descent_direction)
        # --- Vectorized Step-Size Bounding --- 
        neg_mask = descent_direction < 0
        if np.any(neg_mask):
            boundary_limits = -current_beta_array[neg_mask] / descent_direction[neg_mask]
            max_step_limit = np.min(boundary_limits)
        else:
            max_step_limit = float('inf')
        actual_step = min(self.step_size, max_step_limit)
        # --- Updating & Normalizing ---
        updated_beta = current_beta_array + actual_step * descent_direction
        updated_beta = np.clip(updated_beta, 0, 1)
        if np.sum(updated_beta) == 0:
            updated_beta[reference_idx] = 1.0 
        # --- Return ---
        return updated_beta / np.sum(updated_beta)


class AnalyticL2Optimizer(BaseBetaOptimizer):
    """
    AKA: L2-MKL, Closed-form L2, Cauchy-Schwarz Update.
    Goal: Closed-form update for L2-norm MKL (Density).
    Technique: Maintains all kernels in the model. No step size required.
    """
    def update(self, support_vector_weights, Y, kernels_instances_matrix, current_beta_array, **kwargs):
        # --- Value Initialization ---
        weighted_Y = support_vector_weights * Y
        label_outer_product = np.outer(weighted_Y, weighted_Y)
        num_kernels = len(kernels_instances_matrix)
        # --- Vectorized Alignment Calculation ---
        # Note: Non-negative clipping removed for mathematically accurate L2 formulation
        alignment_scores = np.array([np.sum(label_outer_product * km.K) for km in kernels_instances_matrix])
        score_norm = np.linalg.norm(alignment_scores)
        if score_norm == 0:
            return np.ones(num_kernels) / num_kernels 
        updated_beta = alignment_scores / score_norm
        # --- Return ---
        return updated_beta / np.sum(updated_beta)


class ExponentiatedGradientOptimizer(BaseBetaOptimizer):
    """
    AKA: Mirror Descent, Multiplicative Weight Update (MWU), Entropic Descent.
    Goal: Multiplicative update (Mirror Descent). 
    Technique: Highly robust because beta can never become mathematically negative.
    """
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, support_vector_weights, Y, kernels_instances_matrix, current_beta_array, **kwargs):
        # --- Value Initialization ---
        num_kernels = len(kernels_instances_matrix)
        gradient = self.gradient_computer(support_vector_weights, Y, kernels_instances_matrix)
        shifted_gradient = gradient - np.max(gradient)
        # --- Calculating New Beta Array ---
        updated_beta = current_beta_array * np.exp(-self.learning_rate * shifted_gradient)
        sum_beta = np.sum(updated_beta)
        # --- Zero-Division Guard ---
        if sum_beta == 0:
            return np.ones(num_kernels) / num_kernels
        # --- Return ---
        return updated_beta / sum_beta


class ProximalHybridOptimizer(BaseBetaOptimizer):
    """
    AKA: Proximal Gradient, Hybrid L1/L2, Proximal Hybrid.
    Goal: Heuristic combination of L1 and L2 penalties. 
    Technique: Requires a mixing parameter 'rho' (0 = L2 only, 1 = L1 only).
    """
    def __init__(self, mix_ratio=0.5, step_size=0.1):
        self.mix_ratio = mix_ratio
        self.step_size = step_size

    def update(self, support_vector_weights, Y, kernels_instances_matrix, current_beta_array, **kwargs):
        # --- Value Initialization ---
        num_kernels = len(kernels_instances_matrix)
        # --- Calculating Gradient ---
        base_gradient = self.gradient_computer(support_vector_weights, Y, kernels_instances_matrix)
        gradient = base_gradient + (1 - self.mix_ratio) * current_beta_array
        # --- Calculating Gradient Step ---
        gradient_step = current_beta_array - self.step_size * gradient
        # --- Simplex Projection ---
        projection_threshold = self.simplex_projector(gradient_step, num_kernels)
        # --- Calculating New Beta Array ---
        updated_beta = np.maximum(gradient_step - projection_threshold, 0)
        # --- Return ---
        return updated_beta


class KernelTargetAlignmentOptimizer(BaseBetaOptimizer):
    """
    AKA: KTA, Centered Alignment.
    Goal: Optimizes beta based strictly on alignment with the target labels.
    Technique: Ignores SMO alphas entirely. Converges in a single iteration.
    """
    def update(self, support_vector_weights, Y, kernels_instances_matrix, current_beta_array, **kwargs):
        # --- Value Initialization ---
        num_kernels = len(kernels_instances_matrix)
        ideal_target_kernel = np.outer(Y, Y)
        target_norm = np.linalg.norm(ideal_target_kernel, 'fro')
        # --- Vectorized Fast Collection ---
        kernel_inner_products = np.array([np.sum(km.K * ideal_target_kernel) for km in kernels_instances_matrix])
        kernel_norms = np.array([np.linalg.norm(km.K, 'fro') for km in kernels_instances_matrix])
        # --- Safe Division Execution ---
        valid_mask = kernel_norms > 0
        alignments = np.zeros(num_kernels)
        alignments[valid_mask] = kernel_inner_products[valid_mask] / (kernel_norms[valid_mask] * target_norm)
        # --- Alignment Values Clipping ---
        alignments = np.clip(alignments, 0, None)
        # --- Resetting Alignment Values ---
        if np.sum(alignments) == 0:
            return np.ones(num_kernels) / num_kernels
        # --- Return ---
        return alignments / np.sum(alignments)
    

class FrankWolfeOptimizer(BaseBetaOptimizer):
    """
    AKA: Frank-Wolfe (with Line Search)
    Goal: Conditional Gradient method. 
    Technique: Steps toward optimal simplex vertex using surrogate line search estimation.
    """
    def update(self, support_vector_weights, Y, kernels_instances_matrix, current_beta_array, **kwargs):
        # --- Value Initialization ---
        num_kernels = len(kernels_instances_matrix)
        gradient = self.gradient_computer(support_vector_weights, Y, kernels_instances_matrix)
        # --- Finding the Best Vertex ---
        best_vertex_idx = np.argmin(gradient)
        optimal_vertex = np.zeros(num_kernels)
        optimal_vertex[best_vertex_idx] = 1.0
        # --- Local Quadratic Approximation Line Search ---
        direction = optimal_vertex - current_beta_array
        dir_norm_sq = np.sum(direction**2)
        # --- Calculating Step-Size ---
        if dir_norm_sq == 0:
            actual_step_size = 0.0
        else:
            actual_step_size = np.clip(np.dot(-gradient, direction) / dir_norm_sq, 0.0, 1.0)
        # --- Calculating New Beta Array ---
        updated_beta = current_beta_array + actual_step_size * direction
        # --- Return ---
        return updated_beta


class AcceleratedProjectedGradientOptimizer(BaseBetaOptimizer):
    """
    AKA: FISTA / Nesterov (with Adaptive Restart)
    Goal: FISTA-style optimizer. 
    Technique: Uses momentum with adaptive restart to prevent overshooting convergence regions.
    """
    def __init__(self, step_size=0.1):
        self.step_size = step_size
        self.previous_beta = None
        self.prev_momentum_factor = 1.0

    def update(self, support_vector_weights, Y, kernels_instances_matrix, current_beta_array, **kwargs):
        # --- Saving Data ---
        if self.previous_beta is None:
            self.previous_beta = np.copy(current_beta_array)
        # --- Value Initialization ---
        num_kernels = len(kernels_instances_matrix)
        gradient = self.gradient_computer(support_vector_weights, Y, kernels_instances_matrix)
        # --- Adaptive Restart ---
        # If the gradient direction opposes momentum direction, kill momentum.
        if np.dot(gradient, current_beta_array - self.previous_beta) > 0:
            self.prev_momentum_factor = 1.0
        # --- Calculating Nesterov Momentum ---
        curr_momentum_factor = (1.0 + np.sqrt(1.0 + 4.0 * self.prev_momentum_factor**2)) / 2.0
        momentum_ratio = (self.prev_momentum_factor - 1.0) / curr_momentum_factor
        lookahead_beta = current_beta_array + momentum_ratio * (current_beta_array - self.previous_beta)
        gradient_step = lookahead_beta - self.step_size * gradient
        # --- Simplex Projection ---
        projection_threshold = self.simplex_projector(gradient_step, num_kernels)
        # --- Calculating New Beta Array ---
        updated_beta = np.maximum(gradient_step - projection_threshold, 0)
        # --- Saving State ---
        self.previous_beta = np.copy(current_beta_array)
        self.prev_momentum_factor = curr_momentum_factor
        # --- Return ---
        return updated_beta