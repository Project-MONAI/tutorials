import numpy as np
import torch
from sklearn.metrics import roc_auc_score

"""
Derived from:
https://github.com/cellcanvas/album-catalog/blob/main/solutions/copick/compare-picks/solution.py
"""

import numpy as np
import pandas as pd

from scipy.spatial import KDTree


class ParticipantVisibleError(Exception):
    pass


def compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        distance_multiplier: float,
        beta: int,
        weighted=True,
) -> float:
    '''
    F_beta
      - a true positive occurs when
         - (a) the predicted location is within a threshold of the particle radius, and
         - (b) the correct `particle_type` is specified
      - raw results (TP, FP, FN) are aggregated across all experiments for each particle type
      - f_beta is calculated for each particle type
      - individual f_beta scores are weighted by particle type for final score
    '''

    particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }

    weights = {
        'apo-ferritin': 1,
        'beta-amylase': 0,
        'beta-galactosidase': 2,
        'ribosome': 1,
        'thyroglobulin': 2,
        'virus-like-particle': 1,
    }

    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Filter submission to only contain experiments found in the solution split
    split_experiments = set(solution['experiment'].unique())
    submission = submission.loc[submission['experiment'].isin(split_experiments)]

    # Only allow known particle types
    if not set(submission['particle_type'].unique()).issubset(set(weights.keys())):
        raise ParticipantVisibleError('Unrecognized `particle_type`.')

    assert solution.duplicated(subset=['experiment', 'x', 'y', 'z']).sum() == 0
    assert particle_radius.keys() == weights.keys()

    results = {}
    for particle_type in solution['particle_type'].unique():
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

    for experiment in split_experiments:
        for particle_type in solution['particle_type'].unique():
            reference_radius = particle_radius[particle_type]
            select = (solution['experiment'] == experiment) & (solution['particle_type'] == particle_type)
            reference_points = solution.loc[select, ['x', 'y', 'z']].values

            select = (submission['experiment'] == experiment) & (submission['particle_type'] == particle_type)
            candidate_points = submission.loc[select, ['x', 'y', 'z']].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)

            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn

    fbetas = []
    fbeta_weights = []
    particle_types = []
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        fbetas += [fbeta]
        fbeta_weights += [weights.get(particle_type, 1.0)]
        particle_types += [particle_type]
        
    if weighted:
        aggregate_fbeta = np.average(fbetas,weights=fbeta_weights)
    else:
        aggregate_fbeta = np.mean(fbetas)
    
    return aggregate_fbeta, dict(zip(particle_types,fbetas))

def calc_metric(cfg, pp_out, val_df, pre="val"):
    
    particles = cfg.classes
    pred_df = pp_out
    
    solution = val_df.copy()
    solution['id'] = range(len(solution))
    
    submission = pred_df.copy()
    submission['experiment'] = solution['experiment'].unique()[0]
    submission['id'] = range(len(submission))

#     score003 = score(
#         solution.copy(),
#         submission[submission['conf']>0.03].copy(),
#         row_id_column_name = 'id',
#         distance_multiplier=0.5,
#         beta=4)[0]
#     print('score003',score003)

    best_ths = []
    for p in particles:
        sol0a = solution[solution['particle_type']==p].copy()
        sub0a = submission[submission['particle_type']==p].copy()
        scores = []
        ths = np.arange(0,0.5,0.005)
        for c in ths:
            scores += [score(
                        sol0a.copy(),
                        sub0a[sub0a['conf']>c].copy(),
                        row_id_column_name = 'id',
                        distance_multiplier=0.5,
                        beta=4,weighted = False)[0]]
        best_th = ths[np.argmax(scores)]
        best_ths += [best_th]
    
    submission_pp = []
    for th, p in zip(best_ths,particles):
        submission_pp += [submission[(submission['particle_type']==p) & (submission['conf']>th)].copy()]
    submission_pp = pd.concat(submission_pp)
    
    score_pp, particle_scores = score(
        solution[solution['particle_type']!='beta-amylase'].copy(),
        submission_pp.copy(),
        row_id_column_name = 'id',
        distance_multiplier=0.5,
        beta=4)
    
    result = {'score_' + k: v for k,v in particle_scores.items()}
    result['score'] = score_pp
#     print(result)
    return result
# #     if isinstance(pred_df,list):
# #         pred_df,gt_df = pred_df
# #     else:
# #         gt_df = None

#     y_true = val_df['score'].values
#     y_pred = val_data['preds'].cpu().numpy()
#     score = get_score(y_true.flatten(), y_pred.flatten())
# #     print(score)

# #     df['score'] = df['location'].apply(ast.literal_eval)
# #     df['span'] = df['location'].apply(location_to_span)
# #     spans_true = df['span'].values

# #     df_pred = pred_df.copy()
# #     # df_pred['location'] = df_pred['location'].apply(ast.literal_eval)
# #     df_pred['span'] = df_pred['pred_location'].apply(pred_location_to_span)
# #     spans_pred = df_pred['span'].values

# #     score = span_micro_f1(spans_pred, spans_true)

#     if hasattr(cfg, "neptune_run"):
#         cfg.neptune_run[f"{pre}/score/"].log(score, step=cfg.curr_step)
#         print(f"{pre} score: {score:.6}")
# #     else:
# #         return score

# #     if gt_df is not None:
# #         df_pred = gt_df.copy()
# #         df_pred['span'] = df_pred['pred_location'].apply(pred_location_to_span)
# #         spans_pred = df_pred['span'].values

# #         score = span_micro_f1(spans_pred, spans_true)

# #         if hasattr(cfg, "neptune_run"):
# #             cfg.neptune_run[f"{pre}/score_debug/"].log(score, step=cfg.curr_step)
# # #             print(f"{pre} score_debug: {score:.6}")          
#     return score

