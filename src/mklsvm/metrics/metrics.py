import numpy as np
import os
from ..mkl import MKL
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def mkl_svm_evaluation(trained_SVM_model, x_train, y_train, y_test, raw_predicted_Y, primal_obj=None, dual_obj=None, save=False, save_path=None):
    """    
    Evaluates a trained Multiple Kernel Learning SVM model.
    
    #> Parameters:
    trained_SVM_model: An instance of the trained SVM model (after calling SVM.fit())
    y_test: ndarray, shape (N_test,)           : Ground truth labels for test set
    raw_predicted_Y: ndarray, shape (N_test,)   : Raw predicted scores (wx + b) before sign()
    primal_obj: float, optional                 : Calculated primal objective value
    dual_obj: float, optional                   : Calculated dual objective value
    """
    metrics = {}
    # --- Acquiring Necessary Values from the Trained Model ---
    beta_array = trained_SVM_model.MKL_instance.beta_array
    # --- Computing Kernel Sum For Training Data (Shape: N_train x N_train) ---
    untrained_train_MKL = MKL()
    untrained_train_MKL.select_kernels(trained_SVM_model.MKL_instance.selected_kernels, 
                                       trained_SVM_model.MKL_instance.kernel_centering,
                                       trained_SVM_model.MKL_instance.kernel_normalization)
    untrained_train_MKL.kernels_arguments = trained_SVM_model.kernels_arguments
    untrained_train_MKL.kernels_instances_matrix_fit(x_train, untrained_train_MKL.kernels_instances_matrix, untrained_train_MKL.kernels_arguments)
    sum_of_x_train_kernel_matrices = untrained_train_MKL.create_sum_of_kernels_instances_matrix(beta_array, untrained_train_MKL.kernels_instances_matrix, len(x_train))
    # --- Classification Metrics ---
    y_pred = np.sign(raw_predicted_Y)
    y_pred[y_pred == 0] = 1  # Handle boundary edge case
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == -1) & (y_pred == -1))
    fp = np.sum((y_test == -1) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == -1))
    metrics['accuracy'] = (tp + tn) / len(y_test)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0.0
    # --- Structural & MKL-Specific Metrics ---
    # Target Matrix YY^T
    YY_target = np.outer(y_train, y_train)
    # Kernel Target Alignment (Frobenius inner product)
    num = np.sum(sum_of_x_train_kernel_matrices * YY_target)
    denom = np.sqrt(np.sum(sum_of_x_train_kernel_matrices * sum_of_x_train_kernel_matrices) * np.sum(YY_target * YY_target))
    metrics['kernel_target_alignment'] = num / denom if denom > 0 else 0.0
    # Sparsity and Entropy of Weights
    metrics['active_kernels'] = np.sum(beta_array > 1e-5)
    # Normalize entropy (with safety check for single-kernel baselines)
    if np.sum(beta_array) > 0 and len(beta_array) > 1:
        beta_norm = beta_array / np.sum(beta_array)
        metrics['weight_entropy'] = -np.sum(beta_norm * np.log(beta_norm + 1e-12)) / np.log(len(beta_array))
    else:
        metrics['weight_entropy'] = 0.0
    # --- Optimization Sanity Checks ---
    metrics['support_vector_count'] = len(trained_SVM_model.MKL_instance.SMO_model.sv_indices)
    if primal_obj is not None and dual_obj is not None:
        metrics['duality_gap'] = abs(primal_obj - dual_obj)
    else:
        metrics['duality_gap'] = None
    # --- Saving ---
    if save_path is None and save:
        print("Warning: save_path is None. Metrics will not be saved.")
    if save and save_path is not None:
        path = os.path.join(save_path, "mkl-svm.txt")
        with open(path, "w") as f:
            f.write("--- Model Info ---\n")
            for key, value in metrics.items():
                f.write("> " + key + ": " + str(value) + "\n")
            f.close()
    # --- Return ---
    return metrics


def sklearn_semi_mkl_evaluation(trained_SVM_model, x_train, y_train, x_test, y_test, beta=None, save=False, save_path=None):
    """
    Trains an sklearn Precomputed SVC on a multi-kernel matrix structure.
    If beta=None, it uses a uniform baseline [0.20, 0.20, 0.20, 0.20, 0.20, 0].
    """
    # --- Default Beta Handling ---
    if beta is None:
        num_kernels = len(trained_SVM_model.MKL_instance.selected_kernels)
        beta = np.ones(num_kernels) / num_kernels  # Uniform baseline
    # --- Computing Kernel Sum For Training Data (Shape: N_train x N_train) ---
    untrained_train_MKL = MKL()
    untrained_train_MKL.select_kernels(trained_SVM_model.MKL_instance.selected_kernels, 
                                           trained_SVM_model.MKL_instance.kernel_centering,
                                           trained_SVM_model.MKL_instance.kernel_normalization)
    untrained_train_MKL.kernels_arguments = trained_SVM_model.kernels_arguments
    untrained_train_MKL.kernels_instances_matrix_fit(x_train, untrained_train_MKL.kernels_instances_matrix, untrained_train_MKL.kernels_arguments)
    sum_of_x_train_kernel_matrices = untrained_train_MKL.create_sum_of_kernels_instances_matrix(beta, untrained_train_MKL.kernels_instances_matrix, len(x_train))
    # --- Computing Kernel Cross Matrix For Testing Data (Shape: N_train x N_test) ---
    sum_of_x_test_kernel_matrices = np.zeros((len(x_test), len(x_train)))
    args_generator = untrained_train_MKL.kernels_value_yielder(untrained_train_MKL.kernels_arguments)
    # --- Cross Kernel Computation Loop ---
    for loop_beta, kernel, argument in zip(beta, untrained_train_MKL.kernels_instances_matrix, args_generator):
        # compute_cross_K returns shape (n_samples_test, n_samples_train)
        #> scikit-learn precomputed kernel expects shape (n_samples_test, n_samples_train)
        K_cross = kernel.compute_cross_K(x_train, x_test, argument)
        sum_of_x_test_kernel_matrices += loop_beta * K_cross
    # --- Fitting Precomputed Solver ---
    clf = SVC(kernel='precomputed', C=1.0, random_state=42)
    clf.fit(sum_of_x_train_kernel_matrices, y_train)
    # --- Generating Predictions and Process Edge Boundary ---
    raw_predicted_Y = clf.decision_function(sum_of_x_test_kernel_matrices)
    y_pred = np.sign(raw_predicted_Y)
    y_pred[y_pred == 0] = 1
    # --- Computing Pipeline Evaluation Metrics ---
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, pos_label=1, zero_division=0),}
    #> Target Alignment Calculation (Frobenius Product)
    YY_target = np.outer(y_train, y_train)
    num = np.sum(sum_of_x_train_kernel_matrices * YY_target)
    denom = np.sqrt(np.sum(sum_of_x_train_kernel_matrices * sum_of_x_train_kernel_matrices) * np.sum(YY_target * YY_target))
    metrics['kernel_target_alignment'] = num / denom if denom > 0 else 0.0
    metrics['active_kernels'] = np.sum(beta > 1e-5)
    beta_norm = beta / np.sum(beta)
    metrics['weight_entropy'] = -np.sum(beta_norm * np.log(beta_norm + 1e-12)) / np.log(len(beta))
    metrics['support_vector_count'] = int(np.sum(clf.n_support_))
    # --- Saving ---
    if save_path is None and save:
        print("Warning: save_path is None. Metrics will not be saved.")
    if save and save_path is not None:
        path = os.path.join(save_path, "sk-learn-svc.txt")
        with open(path, "w") as f:
            f.write("--- Model Info ---\n")
            for key, value in metrics.items():
                f.write("> " + key + ": " + str(value) + "\n")
            f.close()
    # --- Return ---
    return metrics