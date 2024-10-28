#!/bin/env python
from pathlib import Path  
import pickle
import logging

import laspy
from tqdm import tqdm
import numpy as np

np.set_printoptions(precision=4, suppress=True)

logger = logging.getLogger(__name__)


COMPACT_IDX = {
    0:0,
    2:1,
    5:2,
    6:3,
    9:4,
    17:5}

# reverse the dictionary
COMPACT_IDX_2_ORIGIN_IDX = {v:k for k,v in COMPACT_IDX.items()}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser(description="Remap classification")
    parser.add_argument("input", type=str, help="Path to the input las file")
    parser.add_argument("label_source", type=str, help="Path to the label source las file")
    parser.add_argument("--only_unclassified", action="store_true", help="Only remap unclassified points")
    parser.add_argument("output", type=str, help="Path to the output las file")

    args = parser.parse_args()
    
    input_las = laspy.read(str(args.input))
    reference_las = laspy.read(str(args.label_source))

    # build KDTree for reference las, for each point in input las, find the nearest point in reference las, and assign the classification
    from scipy.spatial import cKDTree
    kdtree = cKDTree(reference_las.xyz)
    dist, idx = kdtree.query(input_las.xyz, k=1)

    sampled_classification = reference_las.classification[idx]

    if args.only_unclassified:
        logger.info("Remap only unclassified points")
        mask = input_las.classification > 1
        input_las.classification[mask] = sampled_classification[mask]
    else:
        logger.info("Remap all points")
        input_las.classification = sampled_classification

    input_las.write(str(args.output))
    
    logger.info(f"Write to {args.output}")
    logger.info('Done')


