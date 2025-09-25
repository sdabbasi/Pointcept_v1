"""
process.py

This script provides utility functions for managing and verifying the correspondence between label files and segment directories
in a dataset split (train/val/test). It is written to replace pointcept-based processed data of semantic segmentation with openpcdet-based processed data.

- 'check': Ensures the number of *_labels.npy files in the source directory matches the number of segment subdirectories in the destination directory for each split entry.
- 'delete': Deletes segment.npy files from the destination directory for each split entry.
- 'verify': Checks that all segment.npy files have been deleted from the destination directory.
- 'copy': Copies each *_labels.npy file from the source directory to the corresponding segment subdirectory in the destination directory, renaming it to segment.npy.

Usage:
    python process.py --split <split_txt_path> --source <source_dir> --dest <destination_dir> --action <action>

Modules used:
    - os: For file and directory operations.
    - tqdm: For progress bars.
    - shutil: For file copying.
    - argparse: For command-line argument parsing.

"""
import os
from tqdm import tqdm

def check_dirs(split_txt_path, source_dir, destination_dir):
    missing = []
    with open(split_txt_path, 'r') as f:
        dir_names = [line.strip().replace('.tfrecord', '') for line in f if line.strip()]
    for name in tqdm(dir_names, desc='Checking directories'):
        path1 = os.path.join(source_dir, name)
        path2 = os.path.join(destination_dir, name)
        if not os.path.isdir(path1) or not os.path.isdir(path2):
            missing.append(name)
    if missing:
        print(f"Number of missing: {len(missing)}")
        raise FileNotFoundError(f"Missing directories in one or both locations: {missing}")
    return dir_names

"""
Checks that the number of *_labels.npy files in source_dir/name matches the number of directories in destination_dir/name for each entry in the split file (train/val/test).
"""
def check_labels_and_segments(split_txt_path, source_dir, destination_dir):
    count_mismatches = []
    with open(split_txt_path, 'r') as f:
        dir_names = [line.strip().replace('.tfrecord', '') for line in f if line.strip()]
    for name in tqdm(dir_names, desc='Checking label/segment counts'):
        path1 = os.path.join(source_dir, name)
        path2 = os.path.join(destination_dir, name)
        if not os.path.isdir(path1) or not os.path.isdir(path2):
            continue
        # Count xxx_labels.npy files in path1
        label_files = [f for f in os.listdir(path1) if f.endswith('_labels.npy')]
        num_label_files = len(label_files)
        # Count segment.npy files inside all subdirectories of path2
        num_segment_files = 0
        for subdir in os.listdir(path2):
            subdir_path = os.path.join(path2, subdir)
            if os.path.isdir(subdir_path):
                segment_file = os.path.join(subdir_path, 'segment.npy')
                if os.path.isfile(segment_file):
                    num_segment_files += 1
        if num_label_files != num_segment_files:
            count_mismatches.append((name, num_label_files, num_segment_files))
    if count_mismatches:
        print(f"Number of count mismatches: {len(count_mismatches)}")
        raise ValueError(f"Count mismatches (name, label_files, segment_files): {count_mismatches}")
    return dir_names

"""
Deletes *_labels.npy files in source_dir/name for each entry in the split file (train/val/test) if previous checks pass.
"""
def delete_segment_files_if_checks_pass(split_txt_path, source_dir, destination_dir):
    with open(split_txt_path, 'r') as f:
        dir_names = [line.strip().replace('.tfrecord', '') for line in f if line.strip()]
    deleted_files = []
    for name in tqdm(dir_names, desc='Deleting segment files'):
        path2 = os.path.join(destination_dir, name)
        if os.path.isdir(path2):
            for subdir in os.listdir(path2):
                subdir_path = os.path.join(path2, subdir)
                if os.path.isdir(subdir_path):
                    segment_file = os.path.join(subdir_path, 'segment.npy')
                    if os.path.isfile(segment_file):
                        try:
                            os.remove(segment_file)
                            deleted_files.append(segment_file)
                        except Exception as e:
                            print(f"Failed to delete {segment_file}: {e}")
    return deleted_files


"""
Checks that all *_labels.npy files in source_dir/name for each entry in the split file have been deleted.
Returns a list of files that still exist (should be empty if deletion was successful).
"""
def check_segments_deleted(split_txt_path, destination_dir):
    with open(split_txt_path, 'r') as f:
        dir_names = [line.strip().replace('.tfrecord', '') for line in f if line.strip()]
    remaining_files = []
    for name in tqdm(dir_names, desc='Verifying segment deletion'):
        path2 = os.path.join(destination_dir, name)
        if os.path.isdir(path2):
            for subdir in os.listdir(path2):
                subdir_path = os.path.join(path2, subdir)
                if os.path.isdir(subdir_path):
                    segment_file = os.path.join(subdir_path, 'segment.npy')
                    if os.path.isfile(segment_file):
                        remaining_files.append(segment_file)
    return remaining_files


"""
Copies each *_labels.npy file from source_dir/name to the corresponding subdirectory in destination_dir/name, renaming it to segment.npy.
The order of *_labels.npy files and subdirectories in path2 is preserved by sorting both lists.
"""
import shutil
def copy_labels_to_segments(split_txt_path, source_dir, destination_dir):
    with open(split_txt_path, 'r') as f:
        dir_names = [line.strip().replace('.tfrecord', '') for line in f if line.strip()]
    copied_files = []
    for name in tqdm(dir_names, desc='Copying label files to segments'):
        path1 = os.path.join(source_dir, name)
        path2 = os.path.join(destination_dir, name)
        if not os.path.isdir(path1) or not os.path.isdir(path2):
            continue
        # Get sorted *_labels.npy files and sorted subdirectories
        label_files = sorted([f for f in os.listdir(path1) if f.endswith('_labels.npy')])
        subdirs = sorted([d for d in os.listdir(path2) if os.path.isdir(os.path.join(path2, d))])
        # Only copy if counts match
        if len(label_files) != len(subdirs):
            print(f"Count mismatch for {name}: {len(label_files)} label files, {len(subdirs)} subdirs")
            continue
        for label_file, subdir in zip(label_files, subdirs):
            src_file = os.path.join(path1, label_file)
            dst_dir = os.path.join(path2, subdir)
            dst_file = os.path.join(dst_dir, 'segment.npy')
            try:
                shutil.copy2(src_file, dst_file)
                copied_files.append((src_file, dst_file))
            except Exception as e:
                print(f"Failed to copy {src_file} to {dst_file}: {e}")
    return copied_files



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process label and segment files.")
    parser.add_argument('--split', required=True, help='Path to split txt file (train/val/test)')
    parser.add_argument('--source', required=True, help='Source directory (contains *_labels.npy files)')
    parser.add_argument('--dest', required=True, help='Destination directory (contains segment.npy files)')
    parser.add_argument('--action', choices=['check', 'delete', 'copy', 'verify'], required=True, help='Action to perform')
    args = parser.parse_args()

    if args.action == 'check':
        # check_dirs(args.split, args.source, args.dest)
        check_labels_and_segments(args.split, args.source, args.dest)
        print('Check passed.')
    elif args.action == 'delete':
        deleted = delete_segment_files_if_checks_pass(args.split, args.source, args.dest)
        print(f'Deleted segment files: {deleted}')
    elif args.action == 'verify':
        remaining = check_segments_deleted(args.split, args.dest)
        if not remaining:
            print('All segment.npy files deleted.')
        else:
            print(f'Segment files still present: {remaining}')
    elif args.action == 'copy':
        copied = copy_labels_to_segments(args.split, args.source, args.dest)
        print(f'Copied label files to segment.npy: {copied}')
    
if __name__ == '__main__':
    main()

# python process.py --split data/waymo_v1.4.3_openpcdet_plus_seg/ImageSets/train.txt --source data/waymo_v1.4.3_openpcdet_plus_seg/waymo_processed_data_v0_5_0 --dest data/waymo_v1.4.3_tmp_copy/processed/training --action check