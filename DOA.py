import numpy as np
from joblib import Parallel, delayed


def calculate_doa_k(proficiency_level, q_matrix, r_matrix, k):
    n_students, n_skills = proficiency_level.shape
    n_questions, _ = q_matrix.shape
    n_attempts = r_matrix.shape[1]
    DOA_k = 0.0
    numerator = 0
    denominator = 0
    delta_matrix = proficiency_level[:, k].reshape(-1, 1) > proficiency_level[:, k].reshape(1, -1)
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for j in question_hask:
        row_vector = (r_matrix[:, j].reshape(1, -1) != -1).astype(int)
        column_vector = (r_matrix[:, j].reshape(-1, 1) != -1).astype(int)
        mask = row_vector * column_vector
        delta_r_matrix = r_matrix[:, j].reshape(-1, 1) > r_matrix[:, j].reshape(1, -1)
        I_matrix = r_matrix[:, j].reshape(-1, 1) != r_matrix[:, j].reshape(1, -1)
        numerator_ = np.logical_and(mask, delta_r_matrix)
        denominator_ = np.logical_and(mask, I_matrix)
        numerator += np.sum(delta_matrix * numerator_)
        denominator += np.sum(delta_matrix * denominator_)

    DOA_k = numerator / denominator
    return DOA_k


def DOA(proficiency_level, q_matrix, r_matrix):
    know_n = q_matrix.shape[1]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k)(proficiency_level, q_matrix, r_matrix, k) for k in range(know_n))
    return np.mean(doa_k_list)


