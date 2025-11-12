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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-trainval',
        help='specify the dataset version')
    args = parser.parse_args()

    root_path = './data/nuscenes'
    version = f'{args.version}'

    nuscenes_data_prep(
        root_path=root_path,
        version=version)
