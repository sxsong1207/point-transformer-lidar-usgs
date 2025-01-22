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

US3D_CLASS_NAMES ={
    0:"Unclassified",
    2:"Ground",
    5:"High Vegetation",
    6:"Building",
    9:"Water",
    17:"Bridge Deck"}

US3D_CLASS_COLOR ={
    0:"#000000",
    2:"#eeeeee",
    5:"#bee784",
    6:"#86868a",
    9:"#6ab8fb",
    17:"#d02420"}

USGS3DEP_CLASS_COLOR ={ i:"#000000" for i in range(0, 256)}
USGS3DEP_CLASS_COLOR.update({
    2:"#eeeeee",
    5:"#bee784",
    6:"#86868a",
    7:"#6ab8fb",
    9:"#6ab8fb",
    17:"#d02420"})
# convert to color to rgb
US3D_CLASS_COLOR = {k: tuple(int(v[i:i+2], 16) for i in (1, 3, 5)) for k,v in US3D_CLASS_COLOR.items()}
USGS3DEP_CLASS_COLOR = {k: tuple(int(v[i:i+2], 16) for i in (1, 3, 5)) for k,v in USGS3DEP_CLASS_COLOR.items()}

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

    in_lasfile = laspy.read(str(input_path))
    # apply COMPACT_IDX_2_ORIGIN_IDX to the classification
    mapped_classifcation = [ COMPACT_IDX_2_ORIGIN_IDX[i] for i in in_lasfile.classification]
    rgb = np.array([USGS3DEP_CLASS_COLOR[i] for i in mapped_classifcation])
    # in_lasfile.header.point_format_id = 3
    # in_lasfile.classification = mapped_classifcation
    # in_lasfile.red = rgb[:,0]
    # in_lasfile.green = rgb[:,1]
    # in_lasfile.blue = rgb[:,2]
    # in_lasfile.write(str(output_path))
        
    out_lasfile =laspy.create(point_format=7, file_version='1.4')
    out_lasfile.header.point_count = in_lasfile.header.point_count
    out_lasfile.header.offset = in_lasfile.header.offset
    out_lasfile.header.scale = in_lasfile.header.scale
    out_lasfile.header.min = in_lasfile.header.min
    out_lasfile.header.max = in_lasfile.header.max
    
    out_lasfile.xyz = in_lasfile.xyz
    out_lasfile.intensity = in_lasfile.intensity
    out_lasfile.return_number = in_lasfile.return_number
    out_lasfile.classification = in_lasfile.classification
    out_lasfile.red = rgb[:,0]
    out_lasfile.green = rgb[:,1]
    out_lasfile.blue = rgb[:,2]
    out_lasfile.write(str(output_path))
    
    logger.info(f"Write to {output_path}")
    logger.info('Done')

