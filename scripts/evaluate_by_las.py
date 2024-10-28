import numpy as np
import laspy
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree

# from util.common_util import intersectionAndUnion
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)

def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate by las")
    parser.add_argument("--gt", type=str, nargs="+", help="Path to the ground truth las file")
    parser.add_argument("--pred", type=str, nargs="+", help="Path to the prediction las file")
    return parser


def extract_aligned_cls(las_1, las_2, tolerance=0.001):
    do_nn_search = False
    num_las_1 = len(las_1.points)
    num_las_2 = len(las_2.points)


    if num_las_1 == num_las_2:
        # sample some points and check their coordinates
        sample_size = 1000
        sample_idx = np.random.randint(0, num_las_1, sample_size)
        sample_1 = np.stack(
            [las_1.x[sample_idx], las_1.y[sample_idx], las_1.z[sample_idx]],
            axis=1,
        )
        sample_2 = np.stack(
            [las_2.x[sample_idx], las_2.y[sample_idx], las_2.z[sample_idx]], axis=1
        )
        dist = np.linalg.norm(sample_1 - sample_2, axis=1)
        if np.all(dist < tolerance):
            do_nn_search = False
        else:
            do_nn_search = True
    else:
        do_nn_search = True
        
    
    raw_1_cls = las_1.classification
    raw_2_cls = las_2.classification
    if do_nn_search:
        xyz_1 = np.stack([las_1.x, las_1.y, las_1.z], axis=1)
        xyz_2 = np.stack([las_2.x, las_2.y, las_2.z], axis=1)

        kdtree_1 = cKDTree(xyz_1)
        nn_dist_2, nn_idx_2 = kdtree_1.query(xyz_2)

        inliers = nn_dist_2 < tolerance

        logger.info(f"[KNN] Number of inliers: {np.sum(inliers)} / {len(inliers)}")
        idx_1 = nn_idx_2[inliers]
        idx_2 = np.where(inliers)[0]

        cls_1 = raw_1_cls[idx_1]
        cls_2 = raw_2_cls[idx_2]
    else:
        cls_1 = raw_1_cls
        cls_2 = raw_2_cls
        
    return cls_1, cls_2

def compute_metrics(gt_cls, pred_cls, num_classes=6, ignore_label=0):
    valid_mask = gt_cls != ignore_label
    gt_cls = gt_cls[valid_mask]
    pred_cls = pred_cls[valid_mask]
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(gt_cls, pred_cls, labels=np.arange(num_classes))
    
    # Initialize variables to store results
    iou_per_class = np.zeros(num_classes)
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_score_per_class = np.zeros(num_classes)
    accuracy_per_class = np.zeros(num_classes)
    
    # Total accuracy
    total_accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    
    for class_id in range(num_classes):
        true_positive = conf_matrix[class_id, class_id]
        false_positive = np.sum(conf_matrix[:, class_id]) - true_positive
        false_negative = np.sum(conf_matrix[class_id, :]) - true_positive
        true_negative = np.sum(conf_matrix) - (true_positive + false_positive + false_negative)
        
        # IoU for the class
        iou = true_positive / (true_positive + false_positive + false_negative) if (true_positive + false_positive + false_negative) > 0 else 0
        iou_per_class[class_id] = iou
        
        # Precision for the class
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        precision_per_class[class_id] = precision

        # Recall for the class
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        recall_per_class[class_id] = recall
        
        # F1 Score for the class
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_score_per_class[class_id] = f1_score
        
        # Accuracy for the class
        accuracy = (true_positive + true_negative) / np.sum(conf_matrix) if np.sum(conf_matrix) > 0 else 0
        accuracy_per_class[class_id] = accuracy

    # return iou_per_class, precision_per_class, accuracy_per_class, recall_per_class, f1_score_per_class, total_accuracy

    print(f"Number of points: {len(pred_cls)} vs {len(gt_cls)}")

    
    mIoU = np.mean(iou_per_class)
    mAcc = np.mean(accuracy_per_class)
    mPrecision = np.mean(precision_per_class)
    mRecall = np.mean(recall_per_class)
    mF1 = np.mean(f1_score_per_class)
    

    print(f"Number of classes: {num_classes}")
    print("result: mIoU/mAcc/mPre/mRecall/mF1 {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f} .".format(mIoU, mAcc, mPrecision, mRecall, mF1))

    for i in range(num_classes):
        print(
            "Class_{} Result: iou/accuracy/precision/recall/F1 {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(
                i, iou_per_class[i], accuracy_per_class[i], precision_per_class[i], recall_per_class[i], f1_score_per_class[i]
            )
        )
    print("Total Accuracy: {:.4f}".format(total_accuracy))
    
def _main():
    logging.basicConfig(level=logging.INFO)
    parser = _get_parser()
    args = parser.parse_args()
    if not args.gt or not args.pred:
        parser.print_help()
        return 1
    
    assert len(args.gt) == len(args.pred), "Number of gt and pred must be the same"
    
    
    tolerance = 0.001  # meter
    
    all_cls_gt = []
    all_cls_pred = []
    for gt_path, pred_path in tqdm(zip(args.gt, args.pred), total=len(args.gt)):    
        las_gt = laspy.read(str(gt_path))
        las_pred = laspy.read(str(pred_path))
        
        cls_gt, cls_pred = extract_aligned_cls(las_gt, las_pred, tolerance=tolerance)
        all_cls_gt.append(cls_gt)
        all_cls_pred.append(cls_pred)
    
    all_cls_gt = np.concatenate(all_cls_gt)
    all_cls_pred = np.concatenate(all_cls_pred)
    # ## Compute Metrics
    compute_metrics(all_cls_gt, all_cls_pred)
    

if __name__ == "__main__":
    import sys
    sys.exit(_main())