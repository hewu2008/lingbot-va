# encoding:utf8
import argparse
import json
from pathlib import Path


TASK_TO_ACTION_TEXT = {
    "put_battery_into_cup": "The robot picks up the battery and places it into the cup.",
    "insert_screw_into_the_hole": "The robot picks up the screw and inserts it into the hole.",
}


def add_action_config_to_episodes(episodes_file: str, output_file: str = None) -> None:
    """Add action_config field to each episode in episodes.jsonl file."""
    input_path = Path(episodes_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {episodes_file}")
    
    if output_file is None:
        output_path = input_path
    else:
        output_path = Path(output_file)
    
    with open(input_path, "r", encoding="utf-8") as f:
        episodes = [json.loads(line.strip()) for line in f if line.strip()]
    
    modified_episodes = []
    for episode in episodes:
        episode_index = episode["episode_index"]
        tasks = episode["tasks"]
        length = episode["length"]
        
        if tasks and tasks[0] in TASK_TO_ACTION_TEXT:
            action_text = TASK_TO_ACTION_TEXT[tasks[0]]
        else:
            action_text = f"Robot performs task: {tasks[0] if tasks else 'unknown'}"
        
        episode["action_config"] = [
            {
                "start_frame": 0,
                "end_frame": length,
                "action_text": action_text,
            }
        ]
        
        modified_episodes.append(episode)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for episode in modified_episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
    
    print(f"Processed {len(modified_episodes)} episodes. Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add action_config field to episodes.jsonl")
    parser.add_argument("--episodes_file", type=str, required=True, help="Path to episodes.jsonl file")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path (default: overwrite input)")
    args = parser.parse_args()
    
    add_action_config_to_episodes(args.episodes_file, args.output_file)