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
    parser.add_argument("output", type=str, help="Path to the output las file")

    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)

    lasfile = laspy.read(str(input_path))
    # apply COMPACT_IDX_2_ORIGIN_IDX to the classification
    origin_classifcation = [ COMPACT_IDX_2_ORIGIN_IDX[i] for i in lasfile.classification]
    lasfile.classification = origin_classifcation
    lasfile.write(str(output_path))
    
    logger.info(f"Write to {output_path}")
    logger.info('Done')

