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

US3D_CLASS_NAMES ={
    0:"Unclassified",
    2:"Ground",
    5:"High Vegetation",
    6:"Building",
    9:"Water",
    17:"Bridge Deck"}

US3D_IDX = {
    0:0,
    2:1,
    5:2,
    6:3,
    9:4,
    17:5}

US3D_IDX_2_US3D_CLASSID = {v:k for k,v in US3D_IDX.items()}

def convert_txt_to_pkl(
    pc3_path: os.PathLike, pkl_path: os.PathLike, cls_path: Optional[os.PathLike] = None
):
    pts = np.genfromtxt(str(pc3_path), delimiter=",", dtype=np.float64)
    if cls_path:
        label = np.genfromtxt(str(cls_path), delimiter=",", dtype=np.uint8).reshape(-1)
        label = np.vectorize(US3D_IDX.get)(label)
        
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
    parser.add_argument(
        "-p", "--portion", type=float, default=0.8, help="train portion of the dataset"
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
    
    all_label_list = sorted(list(raw_dir.glob("*-classification.txt")))
    logger.info(f"Found {len(all_label_list)} label files")
    
    all_ids = [re.sub("_PC-classification.txt", "", i.name) for i in all_label_list]
    np.random.shuffle(all_ids)
    train_ids = all_ids[:int(len(all_ids)*args.portion)]
    val_ids= all_ids[int(len(all_ids)*args.portion):]
    
    category_names_path = dataset_dir / "us3d_names.txt"
    train_ids_path = dataset_dir / "train_ids.txt"
    val_ids_path = dataset_dir / "val_ids.txt"
    trainval_path = dataset_dir / "trainval"
    trainval_path.mkdir(parents=True, exist_ok=True)
    
    if not category_names_path.exists():
        ordered_names = [ US3D_CLASS_NAMES[US3D_IDX_2_US3D_CLASSID[i]] for i in range(len(US3D_IDX))]
        np.savetxt(category_names_path, np.array(ordered_names), fmt="%s")
        
    if not train_ids_path.exists() or not val_ids_path.exists():
        logger.info("Creating train/val split")
        np.savetxt(train_ids_path, train_ids, fmt="%s")
        np.savetxt(val_ids_path, val_ids, fmt="%s")
    else:
        train_ids = np.loadtxt(train_ids_path, dtype=str)
        val_ids = np.loadtxt(val_ids_path, dtype=str)

    logger.info(f"train_ids: {len(train_ids)}")
    logger.info(f"val_ids: {len(val_ids)}")

    for id in tqdm(all_ids, desc="Convert TXT to pkl"):
        pc3_path = raw_dir / f"{id}_PC-reduced.txt"
        cls_path = raw_dir / f"{id}_PC-classification.txt"
        
        pkl_path = trainval_path / f"{id}.pkl"
        if not pkl_path.exists():
            assert pc3_path.exists(), f"{pc3_path} does not exist"
            assert cls_path.exists(), f"{cls_path} does not exist"
            convert_txt_to_pkl(pc3_path, pkl_path, cls_path)
            
    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(_main())
