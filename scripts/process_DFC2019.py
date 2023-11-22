import sys
import os
import numpy as np
from pathlib import Path
import logging
import re
from tqdm import tqdm
from typing import Optional, Tuple, List
import pickle
from collections import OrderedDict
logger = logging.getLogger(__name__)

DFC2019_CLASS_NAMES ={
    0:"Unclassified",
    2:"Ground",
    5:"High Vegetation",
    6:"Building",
    9:"Water",
    17:"Bridge Deck"}

COMPACT_IDX = {
    0:0,
    2:1,
    5:2,
    6:3,
    9:4,
    17:5}
COMPACT_IDX_2_ORIGIN_IDX = {v:k for k,v in COMPACT_IDX.items()}

def convert_txt_to_pkl(
    pc3_path: os.PathLike, pkl_path: os.PathLike, cls_path: Optional[os.PathLike] = None
):
    pts = np.genfromtxt(str(pc3_path), delimiter=",", dtype=np.float64)
    if cls_path:
        label = np.genfromtxt(str(cls_path), delimiter=",", dtype=np.uint8).reshape(-1)
        label = np.vectorize(COMPACT_IDX.get)(label)
        
    else:
        label = None

    pts_mean = np.mean(pts, axis=0)[0:3]
    pts[:, 0:3] -= pts_mean

    pickle.dump(
        (pts.astype(np.float32), label, pts_mean.astype(np.float64)),
        open(str(pkl_path), "wb"),
    )


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Process DFC2019 dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Path to the dataset directory. Contains `raw` subdirectory.(default: %(default)s)",
    )
    return parser

def _main():
    parser = _get_parser()
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    raw_dir = dataset_dir / "raw"

    if not raw_dir.exists():
        logger.error(f"{raw_dir} does not exist")
        parser.print_usage()
        return 1
    
    raw_train_dir = raw_dir / "Train-Track4"
    raw_train_label_dir = raw_dir / "Train-Track4-Truth"
    raw_val_dir = raw_dir / "Validate-Track4"
    raw_val_label_dir = raw_dir / "Validate-Track4-Truth"
    raw_test_dir = raw_dir / "Test-Track4"
    
    if not raw_train_dir.exists():
        logger.error(f"{raw_train_dir} does not exist")
        parser.print_usage()
        return 1
    if not raw_train_label_dir.exists():
        logger.error(f"{raw_train_label_dir} does not exist")
        parser.print_usage()
        return 1
    if not raw_val_dir.exists():
        logger.error(f"{raw_val_dir} does not exist")
        parser.print_usage()
        return 1
    if not raw_val_label_dir.exists():
        logger.error(f"{raw_val_label_dir} does not exist")
        parser.print_usage()
        return 1

    category_names_path = dataset_dir / "dfc2019_names.txt"
    train_ids_path = dataset_dir / "train_ids.txt"
    val_ids_path = dataset_dir / "val_ids.txt"
    test_ids_path = dataset_dir / "test_ids.txt"
    trainval_path = dataset_dir / "trainval"
    trainval_path.mkdir(exist_ok=True)
    test_path = dataset_dir / "test"
    test_path.mkdir(exist_ok=True)
    
    if not category_names_path.exists():
        ordered_names = [ DFC2019_CLASS_NAMES[COMPACT_IDX_2_ORIGIN_IDX[i]] for i in range(len(COMPACT_IDX))]
        np.savetxt(category_names_path, np.array(ordered_names), fmt="%s")

    if not train_ids_path.exists():
        train_ids = [
            re.sub("_PC3.txt", "", i.name) for i in raw_train_dir.glob("*_PC3.txt")
        ]
        np.savetxt(train_ids_path, train_ids, fmt="%s")
    else:
        train_ids = np.loadtxt(train_ids_path, dtype=str)

    if not val_ids_path.exists():
        val_ids = [
            re.sub("_PC3.txt", "", i.name) for i in raw_val_dir.glob("*_PC3.txt")
        ]
        np.savetxt(val_ids_path, val_ids, fmt="%s")
    else:
        val_ids = np.loadtxt(val_ids_path, dtype=str)

    if not test_ids_path.exists():
        test_ids = [
            re.sub("_PC3.txt", "", i.name) for i in raw_test_dir.glob("*_PC3.txt")
        ]
        np.savetxt(test_ids_path, test_ids, fmt="%s")
    else:
        test_ids = np.loadtxt(test_ids_path, dtype=str)

    logger.info(f"train_ids: {len(train_ids)}")
    logger.info(f"val_ids: {len(val_ids)}")
    logger.info(f"test_ids: {len(test_ids)}")

    for id in tqdm(train_ids, desc="Convert Train pkl"):
        pc3_path = raw_train_dir / f"{id}_PC3.txt"
        cls_path = raw_train_label_dir / f"{id}_CLS.txt"
        pkl_path = trainval_path / f"{id}.pkl"
        if not pkl_path.exists():
            assert pc3_path.exists(), f"{pc3_path} does not exist"
            assert cls_path.exists(), f"{cls_path} does not exist"
            convert_txt_to_pkl(pc3_path, pkl_path, cls_path)

    for id in tqdm(val_ids, desc="Convert Validate pkl"):
        pc3_path = raw_val_dir / f"{id}_PC3.txt"
        cls_path = raw_val_label_dir / f"{id}_CLS.txt"
        pkl_path = trainval_path / f"{id}.pkl"
        if not pkl_path.exists():
            assert pc3_path.exists(), f"{pc3_path} does not exist"
            assert cls_path.exists(), f"{cls_path} does not exist"
            convert_txt_to_pkl(pc3_path, pkl_path, cls_path)

    for id in tqdm(test_ids, desc="Convert Test pkl"):
        pc3_path = raw_test_dir / f"{id}_PC3.txt"
        pkl_path = test_path / f"{id}.pkl"
        if not pkl_path.exists():
            assert pc3_path.exists(), f"{pc3_path} does not exist"
            convert_txt_to_pkl(pc3_path, pkl_path)

    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(_main())
