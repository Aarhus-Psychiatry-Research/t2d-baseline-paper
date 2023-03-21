import re


def feature_name_to_readable(full_string: str) -> str:
    """Takes a feature name and returns a human readable version of it."""
    if "within" not in full_string:
        output_string = parse_static_feature(full_string)
    else:
        output_string = parse_temporal_feature(full_string)

    return output_string


def parse_static_feature(full_string: str) -> str:
    """Takes a static feature name and returns a human readable version of it."""
    feature_name = full_string.replace("pred_", "")
    feature_capitalised = feature_name[0].upper() + feature_name[1:]
    return feature_capitalised


def parse_temporal_feature(full_string: str) -> str:
    """Takes a temporal feature name and returns a human readable version of it."""
    feature_name = re.findall(r"pred_(.*)?_within", full_string)[0]

    feature_name_mappings = {
        "hba1c": "HbA1c",
    }

    if feature_name in feature_name_mappings:
        feature_name = feature_name_mappings[feature_name]

    lookbehind = re.findall(r"within_(.*)?_days", full_string)[0]

    resolve_multiple = re.findall(r"days_(.*)?_fallback", full_string)[0]

    output_string = f"{lookbehind}-day {resolve_multiple} {feature_name}"
    return output_string
