import argparse
import glob
import numpy as np
import pyarrow.parquet as pq


def compute_action_stats(dataset_dir: str) -> None:
    parquet_files = glob.glob(f'{dataset_dir}/data/**/*.parquet', recursive=True)
    
    if not parquet_files:
        print(f"No parquet files found in {dataset_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_actions = []
    
    for i, file_path in enumerate(parquet_files):
        if (i + 1) % 10 == 0:
            print(f"Processing file {i+1}/{len(parquet_files)}...")
        
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        actions = np.stack(df['action'].values)
        all_actions.append(actions)
    
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"\nTotal frames: {all_actions.shape[0]}")
    print(f"Action dimension: {all_actions.shape[1]}")
    
    q01 = np.quantile(all_actions, 0.01, axis=0)
    q99 = np.quantile(all_actions, 0.99, axis=0)
    
    print("\n=== q01 (1st percentile) ===")
    print("q01 = [")
    for i, val in enumerate(q01):
        print(f"    {val},  # dim={i}")
    print("]")
    
    print("\n=== q99 (99th percentile) ===")
    print("q99 = [")
    for i, val in enumerate(q99):
        print(f"    {val},  # dim={i}")
    print("]")
    
    print("\n=== Summary ===")
    print("  dim | q01      | q99      | range")
    print("------|----------|----------|------")
    for i in range(len(q01)):
        print(f"  {i:3d} | {q01[i]:8.4f} | {q99[i]:8.4f} | {q99[i]-q01[i]:6.4f}")
    
    print("\n=== Dimension Mapping (HDF5 order) ===")
    print("  0-13: arm/position (双臂关节, 左0-6, 右7-13)")
    print(" 14-15: base/velocity (底盘速度)")
    print(" 16-17: effector/position (夹爪, 左16, 右17)")
    print(" 18-31: end/position (双臂EEF, 左18-24, 右25-31)")
    print(" 32-33: head/position (头部)")
    print(" 34-36: waist/position (腰部)")
    
    return q01, q99


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute action q01 and q99 statistics from parquet files')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to LeRobot dataset directory')
    args = parser.parse_args()
    
    compute_action_stats(args.dataset_dir)