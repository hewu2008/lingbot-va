# encoding:utf8
import os
import argparse
import zlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import h5py
import numpy as np
import cv2
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import write_task, write_episode, write_stats

def show_lerobot_record_data(local_root, show_field_ranges=False):
    dataset = LeRobotDataset(local_root, video_backend="pyav")
    print("LeRobot record data:", dataset)

    sample = dataset[0]
    print("observation.images.top shape:", sample['observation.images.top'].shape)
    print("observation.images.right shape:", sample['observation.images.right'].shape)
    print("observation.state:", sample['observation.state'])
    print("action:", sample['action'])
    print("timestamp:", sample['timestamp'])  # episode timestamp in seconds
    print("frame_index:", sample['frame_index'])  # episode frame number 
    print("index:", sample['index'])  # global frame index
    print("episode_index:", sample['episode_index'])
    print("task_index:", sample['task_index'])

    if show_field_ranges:
        timestamps = []
        frame_indices = []
        episode_indices = []
        indices = []
        task_indices = []
        
        for i in tqdm(range(len(dataset)), desc="Iterating through dataset"):
            s = dataset[i]
            timestamps.append(float(s['timestamp']))
            frame_indices.append(int(s['frame_index']))
            episode_indices.append(int(s['episode_index']))
            indices.append(int(s['index']))
            task_indices.append(int(s['task_index']))

            episode_index = int(s['episode_index'])
            if episode_index > 1:
                break
        
        print("\n--- Dataset field ranges ---")
        print(f"timestamp: min={min(timestamps):.3f}, max={max(timestamps):.3f}")
        print(f"frame_index: min={min(frame_indices)}, max={max(frame_indices)}")
        print(f"episode_index: min={min(episode_indices)}, max={max(episode_indices)}")
        print(f"index: min={min(indices)}, max={max(indices)}")
        print(f"task_index: min={min(task_indices)}, max={max(task_indices)}")

def _bytes_from_hdf5_item(item: Any) -> bytes:
    if hasattr(item, "tobytes"):
        return item.tobytes()
    return bytes(item)

def _decode_encoded_image(buf: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return img

def _decode_zlib_raw(buf: bytes, orig_shape: Tuple[int, ...], orig_dtype: Optional[np.dtype]) -> Optional[np.ndarray]:
    try:
        decompressed = zlib.decompress(buf)
    except Exception:
        return None
    if orig_dtype is None or not orig_shape:
        return np.frombuffer(decompressed, dtype=np.uint8)
    try:
        return np.frombuffer(decompressed, dtype=orig_dtype).reshape(orig_shape)
    except Exception:
        return None

def find_hdf5_files(root_dir: str) -> List[Path]:
    """Find all HDF5 files in directory"""
    hdf5_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".hdf5"):
                hdf5_files.append(Path(root) / f)
    print(f"Found {len(hdf5_files)} HDF5 files")
    return hdf5_files

def _recursive_get_datasets(group, path="") -> List[str]:
    """Recursively get all dataset paths from HDF5 group"""
    datasets = []
    for name, obj in group.items():
        current_path = f"{path}/{name}" if path else name
        if isinstance(obj, h5py.Dataset):
            datasets.append(current_path)
        elif isinstance(obj, h5py.Group):
            datasets.extend(_recursive_get_datasets(obj, current_path))
    return datasets

class HDF5EpisodeReader:
    def __init__(self, hdf5_path: Path):
        if not hdf5_path.exists():
            raise FileNotFoundError(f"Dataset does not exist at: {hdf5_path}")
        self.hdf5_path = hdf5_path
        self.file = h5py.File(str(hdf5_path), "r")
        self.file_name = Path(hdf5_path).stem
        self.fps = self._read_fps()
        self.camera_paths = sorted(self._resolve_camera_paths())
        self.camera_shapes = self._resolve_camera_shapes()
        self.state_paths = sorted(self._resolve_state_paths())
        self.action_paths = sorted(self._resolve_action_paths())
        self.timestamp_path = self._resolve_timestamp_path()
        self.task_name = self._resolve_task_name()

    def _resolve_task_name(self) -> str:
        """Resolve task name from HDF5 file attributes"""
        # Try to read from root attributes
        assert "task_name" in self.file.attrs
        return str(self.file.attrs["task_name"])

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

    def _read_fps(self) -> float:
        fps = None
        if "timestamp" in self.file:
            fps = self.file["timestamp"].attrs.get("rate_hz", None)
        if fps is None:
            return 30.0
        try:
            return float(fps)
        except Exception:
            return 30.0

    def _resolve_camera_paths(self) -> List[str]:
        base = "observation/images"
        group = self.file[base]
        return _recursive_get_datasets(group, base)

    def _resolve_state_paths(self) -> List[str]:
        base = "observation/state"
        group = self.file[base]
        return _recursive_get_datasets(group, base)

    def _resolve_action_paths(self) -> List[str]:
        base = "action"
        group = self.file[base]
        return _recursive_get_datasets(group, base)

    def _resolve_timestamp_path(self) -> str:
        return "timestamp/t"

    def _resolve_camera_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Resolve camera image shapes from HDF5 attributes"""
        shapes = {}
        for camera_path in self.camera_paths:
            obj = self.file[camera_path]
            if "orig_shape" in obj.attrs:
                shape = tuple(obj.attrs["orig_shape"])
                shapes[camera_path] = [int(v) for v in shape]
        return shapes
    
    def get_shape_by_dataset(self, dataset_path: str) -> Tuple[int, ...]:
        return self.file[dataset_path].shape

    def get_camera_name_by_path(self, path: str) -> str:
        return path.split("/")[-2]
    
    def get_state_name_by_path(self, path: str) -> str:
        return ".".join(path.split("/")[-2:])
    
    def get_action_name_by_path(self, path: str) -> str:
        return ".".join(path.split("/")[-2:])

    def get_frame(self, camera_path: str, idx: int) -> Optional[np.ndarray]:
        if camera_path not in self.camera_paths:
            return None
        dset = self.file[camera_path]
        meta = dset.attrs
        item = dset[idx]
        encoded_format = meta["encoded_format"]
        orig_shape = meta["orig_shape"]
        orig_dtype = meta["orig_dtype"]

        if encoded_format in (b"jpeg", "jpeg", b"png", "png"):
            buf = _bytes_from_hdf5_item(item)
            img = _decode_encoded_image(buf)
            if img is not None:
                return img
            if orig_shape and orig_dtype is not None:
                try:
                    return np.frombuffer(buf, dtype=orig_dtype).reshape(orig_shape)
                except Exception:
                    return None
            return None
        try:
            return np.array(item)
        except Exception:
            return None

class HDF5ToLeRobotConverter:
    """Converter class to convert HDF5 files to LeRobotDataset format"""

    def __init__(self, output_root: str, repo_id: str, robot_type: str = "Zerith_H1"):
        self.output_root = output_root
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.dataset = None  # Shared LeRobotDataset instance for all episodes
        self.episode_index = 0  # Track current episode index

    def get_lerobot_features(self, reader: HDF5EpisodeReader) -> dict:
        """Build features dictionary for LeRobotDataset from HDF5 content"""
        features = {}
        self._add_camera_features(features, reader)
        self._add_state_features(features, reader)
        self._add_action_features(features, reader)
        return features

    def _add_camera_features(self, features: dict, reader: HDF5EpisodeReader) -> None:
        """Add camera image features to features dict"""
        for camera_path in reader.camera_paths:
            if camera_path.endswith("color"):
                camera_name = reader.get_camera_name_by_path(camera_path)
                camera_shape = reader.camera_shapes.get(camera_path)
                feature_key = f"observation.images.{camera_name}"
                features[feature_key] = {
                    "dtype": "video",
                    "shape": camera_shape,
                    "names": ["height", "width", "channels"],
                }

    def _add_state_features(self, features: dict, reader: HDF5EpisodeReader) -> None:
        """Merge all state paths into single observation.state feature"""
        if not reader.state_paths:
            return
        
        total_dim = 0
        state_names = []
        for state_path in reader.state_paths:
            state_shape = reader.get_shape_by_dataset(state_path)
            state_name = reader.get_state_name_by_path(state_path)
            assert len(state_shape) == 2, f"State shape must be 2D, got {len(state_shape)}D for {state_path}"
            total_dim += state_shape[1]
            state_names.append(state_name)
        
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (total_dim,) if total_dim > 0 else (),
            "names": state_names,
        }

    def _add_action_features(self, features: dict, reader: HDF5EpisodeReader) -> None:
        """Merge all action paths into single action feature"""
        if not reader.action_paths:
            return
        
        total_dim = 0
        action_names = []
        for action_path in reader.action_paths:
            action_shape = reader.get_shape_by_dataset(action_path)
            action_name = reader.get_action_name_by_path(action_path)
            assert len(action_shape) == 2, f"Action shape must be 2D, got {len(action_shape)}D for {action_path}"
            total_dim += action_shape[1]
            action_names.append(action_name)
        
        features["action"] = {
            "dtype": "float32",
            "shape": (total_dim,) if total_dim > 0 else (),
            "names": action_names,
        }

    def _create_tasks_and_stats(self, output_path: Path) -> None:
        """Create tasks.jsonl and stats.json files if they don't exist"""
        meta_path = output_path / "meta"
        
        # Create tasks.jsonl if not exists
        if not (meta_path / "tasks.jsonl").exists():
            print("Creating tasks.jsonl...")
            write_task(0, "Grab the red cube and place it on the plate", output_path)

    def _init_dataset(self, reader: HDF5EpisodeReader) -> None:
        """Initialize LeRobotDataset on first HDF5 file"""
        output_path = Path(self.output_root) / self.repo_id
        
        # Clean up existing output directory
        shutil.rmtree(output_path, ignore_errors=True)
        
        features = self.get_lerobot_features(reader)
        
        # Use LeRobotDataset.create to create a new dataset from scratch
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=int(reader.fps),
            features=features,
            root=output_path,
            robot_type=self.robot_type,
            use_videos=True,
        )
        
        # Create additional metadata files (tasks.parquet and stats.json)
        # self._create_tasks_and_stats(output_path)
        
        print(f"Created dataset with features: {list(features.keys())}")

    def convert_single_file(self, hdf5_path: Path) -> None:
        """Convert single HDF5 file to an episode in LeRobotDataset"""
        reader = None
        
        try:
            reader = HDF5EpisodeReader(hdf5_path)
            print(f"\nProcessing episode {self.episode_index}: {hdf5_path.name}")
            print(f"FPS: {reader.fps}")
            print(f"Task Name: {reader.task_name}")
            print("Cameras Paths:")
            for path in reader.camera_paths: print(f"  - {path}")
            print("States Paths:")
            for path in reader.state_paths: print(f"  - {path}")
            print("Actions Paths:")
            for path in reader.action_paths: print(f"  - {path}")
            
            # Initialize dataset on first episode
            if self.dataset is None:
                self._init_dataset(reader)
            
            # Get total frames
            n_frames = int(reader.file[reader.timestamp_path].shape[0])
            print(f"Total frames: {n_frames}")
            
            # Read frame by frame and add to dataset
            print("Reading and saving frames...")
            for frame_idx in tqdm(range(n_frames), desc=f"Processing episode {self.episode_index} frames"):
                frame = {}
                
                # Add images
                for camera_path in reader.camera_paths:
                    # Only process color cameras
                    if not camera_path.endswith("color"):
                        continue
                    camera_name = reader.get_camera_name_by_path(camera_path)
                    feature_key = f"observation.images.{camera_name}"
                    img = reader.get_frame(camera_path, frame_idx)
                    if img is not None:
                        # Ensure BGR format (OpenCV default)
                        if img.ndim == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif img.shape[2] == 4:
                            img = img[:, :, :3]
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        frame[feature_key] = img
                
                # Add merged state (concatenate all state paths on axis=0)
                if reader.state_paths:
                    state_parts = [reader.file[state_path][frame_idx] for state_path in reader.state_paths]
                    state_data = np.concatenate(state_parts, axis=0) if len(state_parts) > 1 else state_parts[0]
                    frame["observation.state"] = state_data.astype(np.float32).flatten()
                
                # Add action (if exists)
                if reader.action_paths:
                    action_parts = [reader.file[action_path][frame_idx] for action_path in reader.action_paths]
                    action_data = np.concatenate(action_parts, axis=0) if len(action_parts) > 1 else action_parts[0]
                    frame["action"] = action_data.astype(np.float32).flatten()
                
                # Add frame to dataset (timestamp, frame_index, index, episode_index are auto-generated by add_frame)
                self.dataset.add_frame(frame, reader.task_name)
            
            # Save episode to disk
            print("Saving episode...")
            self.dataset.save_episode()
            self.episode_index += 1
            print(f"Completed episode {self.episode_index - 1}: {hdf5_path.name}")
        finally:
            if reader:
                reader.close()

    def finalize(self) -> None:
        """Finalize the dataset after processing all episodes"""
        if self.dataset is not None:
            print("\nFinalizing dataset...")
            print(f"Dataset created successfully with {self.episode_index} episodes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 files to LeRobotDataset")
    parser.add_argument("--hdf5_dir", type=str, required=True, help="Directory containing HDF5 files")
    parser.add_argument("--output_root", type=str, required=True, help="Output root directory")
    parser.add_argument("--repo_id", type=str, default="converted_hdf5_dataset", help="Dataset name")
    parser.add_argument("--robot_type", type=str, default="Zerith_H1", help="Robot type")
    args = parser.parse_args()
    
    hdf5_files = find_hdf5_files(args.hdf5_dir)
    
    if not hdf5_files:
        print("No HDF5 files found")
        exit(1)
    
    converter = HDF5ToLeRobotConverter(args.output_root, args.repo_id, args.robot_type)
    
    # Process HDF5 files one by one (each file becomes an episode)
    for hdf5_path in hdf5_files:
        converter.convert_single_file(hdf5_path)
    
    # Finalize the dataset after all episodes are processed
    converter.finalize()
    
    print("\nAll tasks completed successfully")