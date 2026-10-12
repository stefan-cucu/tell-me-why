import copy


def apply_perturbation_diff(original, diff):
    modified = copy.deepcopy(original)
    for category, items in diff.get("add", {}).items():
        if category not in modified:
            modified[category] = []
        if not isinstance(modified[category], list):
            continue
        values = items if isinstance(items, list) else [items]
        existing = set(modified[category])
        modified[category].extend(value for value in values if value not in existing)
    for category, edits in diff.get("edit", {}).items():
        if category not in modified or not isinstance(modified[category], list):
            continue
        if not isinstance(edits, list):
            continue
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            old_value = edit.get("old")
            new_value = edit.get("new")
            if old_value is None or new_value is None:
                continue
            for index, value in enumerate(modified[category]):
                if value == old_value:
                    modified[category][index] = new_value
                    break
    return modified
