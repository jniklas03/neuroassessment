"""Subject-level group assignments shared by recording and review."""

import csv
from pathlib import Path

GROUPS = ("control", "disease")


def read_groups(data_dir):
    path = Path(data_dir) / "subject_groups.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"subject_name", "group"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} needs subject_name and group columns")
        groups = {}
        for row in reader:
            name = row["subject_name"].strip()
            group = row["group"].strip().lower()
            if not name or group not in ("", *GROUPS):
                raise ValueError(f"Invalid subject/group entry in {path}: {row}")
            if name in groups:
                raise ValueError(f"Duplicate subject {name!r} in {path}")
            groups[name] = group
    return groups


def read_dominant_hands(data_dir):
    path = Path(data_dir) / "subject_groups.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    hands = {}
    for row in rows:
        hand = (row.get("dominant_hand") or "").strip().lower()
        if hand not in ("", "left", "right"):
            raise ValueError("dominant_hand must be left or right")
        hands[row["subject_name"].strip()] = hand
    return hands


def write_groups(data_dir, groups, dominant_hands=None):
    directory = Path(data_dir)
    hands = read_dominant_hands(directory) if dominant_hands is None else dominant_hands
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "subject_groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_name", "group", "dominant_hand"])
        writer.writerows((name, group, hands.get(name, "")) for name, group in sorted(groups.items()))


def get_dominant_hand(subject_name, data_dir, dominant_hand=None):
    groups = read_groups(data_dir)
    hands = read_dominant_hands(data_dir)
    hand = dominant_hand or hands.get(subject_name)
    while not hand:
        answer = input(f"Dominant hand for {subject_name} (left/right): ").strip().lower()
        if answer in ("left", "right"):
            hand = answer
        else:
            print("Enter left or right.")
    hand = hand.strip().lower()
    if hand not in ("left", "right"):
        raise ValueError("dominant_hand must be left or right")
    hands[subject_name] = hand
    groups.setdefault(subject_name, "")
    write_groups(data_dir, groups, hands)
    return hand


def get_subject_group(subject_name, data_dir, group=None):
    groups = read_groups(data_dir)
    existing = groups.get(subject_name)
    if group is None:
        group = existing
        while not group:
            answer = input(f"Group for {subject_name} (control/disease): ").strip().lower()
            if answer in GROUPS:
                group = answer
            else:
                print("Enter control or disease.")
    group = group.strip().lower()
    if group not in GROUPS:
        raise ValueError("group must be control or disease")
    if existing and existing != group:
        raise ValueError(f"{subject_name} is already assigned to {existing}; edit subject_groups.csv to change it")
    groups[subject_name] = group
    write_groups(data_dir, groups)
    return group
