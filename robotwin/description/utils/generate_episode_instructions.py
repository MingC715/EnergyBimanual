import itertools
import json
from pathlib import Path
import random
import re


DESCRIPTION_ROOT = Path(__file__).resolve().parents[1]


def filter_instructions(instructions, episode):
    keys = {key.strip("{}") for key in episode}
    arms = {key for key in keys if len(key) == 1 and "a" <= key <= "z"}
    random.shuffle(instructions)
    valid = []
    for instruction in instructions:
        placeholders = set(re.findall(r"{([^}]+)}", instruction))
        if placeholders == keys or (arms and placeholders == keys - arms):
            valid.append(instruction)
    return valid


def render_instruction(template, episode, split):
    for placeholder, value in episode.items():
        key = placeholder.strip("{}")
        file = DESCRIPTION_ROOT / "objects_description" / (value + ".json")
        if file.is_file():
            descriptions = json.loads(file.read_text(encoding="utf-8"))
            options = descriptions.get(split) or descriptions["seen"]
            text = "the " + random.choice(options)
        elif "/" in value or "\\" in value:
            raise FileNotFoundError(file)
        elif len(key) == 1 and "a" <= key <= "z":
            text = f"the {value} arm"
        else:
            text = value
        template = template.replace("{" + key + "}", text)
    return template


def generate_episode_descriptions(task_name, episodes, max_descriptions=1000000):
    file = DESCRIPTION_ROOT / "task_instruction" / (task_name + ".json")
    templates = json.loads(file.read_text(encoding="utf-8"))
    output = []
    for index, episode in enumerate(episodes):
        # Both template splits share a random number stream.
        splits = {name: filter_instructions(templates.get(name, []), episode) for name in ("seen", "unseen")}
        if not any(splits.values()):
            continue
        entry = {"episode_index": index}
        for name, choices in splits.items():
            pool = itertools.islice(itertools.cycle(choices), max(0, max_descriptions))
            entry[name] = [render_instruction(template, episode, name) for template in pool]
        output.append(entry)
    return output
