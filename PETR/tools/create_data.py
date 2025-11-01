# Copyright (c) OpenMMLab. All rights reserved.

import argparse
from data_converter import nuscenes_converter as nuscenes_converter


def nuscenes_data_prep(root_path,
                       version,
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
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

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
    required=False,
    help='specify the dataset version')
args = parser.parse_args()

if __name__ == '__main__':
    if args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            version=train_version)
    
    elif args.version == 'v1.0':
       train_version = f'{args.version}-trainval'
       nuscenes_data_prep(
           root_path=args.root_path,
           version=train_version)
       
        # test_version = f'{args.version}-test'
        # nuscenes_data_prep(
        #     root_path=args.root_path,
        #     info_prefix=args.extra_tag,
        #     version=test_version,
        #     out_dir=args.out_dir,
        #     max_sweeps=args.max_sweeps)
