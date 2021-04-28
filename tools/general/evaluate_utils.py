import numpy as np

def get_scores(pred_tags, gt_tags):
    if len(pred_tags) == 0 and len(gt_tags) == 0:
        return 100, 100, 100
    elif len(pred_tags) == 0 or len(gt_tags) == 0:
        return 0, 0, 0
    
    pred_tags = np.asarray(pred_tags)
    gt_tags = np.asarray(gt_tags)

    precision = pred_tags[:, np.newaxis] == gt_tags[np.newaxis, :]
    recall = gt_tags[:, np.newaxis] == pred_tags[np.newaxis, :]
    
    precision = np.sum(precision) / len(precision) * 100
    recall = np.sum(recall) / len(recall) * 100
    
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1_score