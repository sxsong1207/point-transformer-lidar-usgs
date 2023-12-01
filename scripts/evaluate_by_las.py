import numpy as np
import laspy
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree

from util.common_util import intersectionAndUnion
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

def compute_metrics(gt_cls, pred_cls):
    num_classes = max(gt_cls.max(), pred_cls.max()) + 1
    ignore_index = 0
    print(f"Number of points: {len(pred_cls)} vs {len(gt_cls)}")

    intersection, union, target = intersectionAndUnion(
        pred_cls, gt_cls, num_classes, ignore_index=ignore_index
    )

    labels, num_labels = np.unique(gt_cls, return_counts=True)

    print("intersection=", intersection)
    print("union=", union)
    print("target=", target)
    print("num_labels=", num_labels)

    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)

    print(f"Number of classes: {num_classes}")
    print(iou_class.shape)
    print("result: mIoU/mAcc {:.4f}/{:.4f} .".format(mIoU, mAcc))

    for i in range(len(labels)):
        print(f"Class {labels[i]}: {num_labels[i]} samples")
        print(
            "Class_{} Result: iou/accuracy {:.4f}/{:.4f}".format(
                labels[i], iou_class[i], accuracy_class[i]
            )
        )
    
def _main():
    logging.basicConfig(level=logging.INFO)
    parser = _get_parser()
    args = parser.parse_args()
    
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