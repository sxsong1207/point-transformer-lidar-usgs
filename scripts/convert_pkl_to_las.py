import numpy as np
import laspy
from pathlib import Path
import pickle
from tqdm import tqdm
from scripts.stitching_tiles import write_las

def convert_pkl_to_las(in_pkl_path, out_las_path, in_label_path = None):
  Path(out_las_path).parent.mkdir(parents=True, exist_ok=True)
  
  data, label, xyz_offset = pickle.load(open(in_pkl_path, "rb"))[:3]
  data = data.astype(np.float64)
  data[:,:3] += xyz_offset
  
  if in_label_path:
    label = np.fromfile(in_label_path, dtype=np.uint8)
  write_las(data[:,:3], data[:,3], data[:,4], label, xyz_offset, out_las_path)


def _get_parser():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("in_path", type=str, help="Path to a pkl file or a directory of pkl files")
  parser.add_argument("out_dir", type=str, help="Path to a directory to save las files")
  return parser

def _main():
  parser = _get_parser()  
  args = parser.parse_args()
  in_path = Path(args.in_path)
  out_dir = Path(args.out_dir)
  assert out_dir.is_dir() or not out_dir.exists(), "out_path must be a directory if in_path is a directory"
  out_dir.mkdir(parents=True, exist_ok=True)
  
  if in_path.is_dir():
    in_paths = list(in_path.glob("*.pkl"))  
  elif in_path.is_file():
    in_paths = [in_path]
  
  for in_pkl_path in tqdm(in_paths,desc="Reading pkl"):
    out_las_path = out_dir / (in_pkl_path.stem + ".las")
    convert_pkl_to_las(in_pkl_path, out_las_path)
  return 0

if __name__ == "__main__":
  import sys
  sys.exit(_main())
