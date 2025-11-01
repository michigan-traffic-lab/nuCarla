# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

from data_converter import nuscenes_converter as nuscenes_converter
import argparse
import sys
sys.path.append('.')


def nuscenes_data_prep(root_path,
                       version,
                       out_dir,
                       info_prefix='nuscenes',
                       max_sweeps=10):
    
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, info_prefix, version=version, max_sweeps=max_sweeps)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    help='specify the dataset version')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/nuscenes',
    help='name of info pkl')
args = parser.parse_args()

if __name__ == '__main__':
    if args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            version=train_version,
            out_dir=args.out_dir)
        
    elif args.version == 'v1.0':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            version=train_version,
            out_dir=args.out_dir)
        
        # test_version = f'{args.version}-test'
        # nuscenes_data_prep(
        #     root_path=args.root_path,
        #     version=test_version,
        #     out_dir=args.out_dir)
        
    else:
        raise TypeError("Unsupported nuscenes version.")