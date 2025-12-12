import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

# Reuse script components from the fine-tuning prompt processor
from json_data.transform_prompts import (
    get_info,
    load_json,
    normalize_course_code,
    DEGREE_ALIAS_MAP,
)

COURSES_PATH = Path("json_data/courses.json")
DEGREES_PATH = Path("json_data/degrees.json")

# Regex to find course codes
COURSE_CODE_RE = re.compile(
    r"\b(?P<prefix>cs|ds)\s*(?P<number>\d{4})\b",
    flags=re.IGNORECASE,
)

DEGREE_SYNONYMS: Dict[str, List[str]] = {
    "BS_CS": [
        "bs computer science",
        "b.s. computer science",
        "bachelor of science in computer science",
        "bs in computer science",
        "computer science major",
        "cs major",
        "undergrad computer science",
        "undergraduate computer science",
        "bs cs",
        "bscs",
    ],
    "BS_DS": [
        "bs data science",
        "b.s. data science",
        "bachelor of science in data science",
        "bs in data science",
        "data science major",
        "ds major",
        "undergrad data science",
        "undergraduate data science",
        "bs ds",
        "bs datasci",
    ],
}


def extract_course_codes(text: str) -> Set[str]:
    codes: Set[str] = set()

    for match in COURSE_CODE_RE.finditer(text):
        prefix = match.group("prefix")
        number = match.group("number")
        raw_code = f"{prefix}{number}"
        norm = normalize_course_code(raw_code)
        codes.add(norm)

    return codes


def extract_degree_id(text: str, manual_degree: Optional[str] = None) -> Optional[str]:
    """
    Return the canonical degree ID used in training ("BS_CS" or "BS_DS"),
    based on the user's text and/or a manually specified value.
    """
    text_l = text.lower()

    if manual_degree:
        md = manual_degree.strip().upper()
        if md in DEGREE_ALIAS_MAP:
            return md

    for degree_id, phrases in DEGREE_SYNONYMS.items():
        for phrase in phrases:
            if phrase in text_l:
                return degree_id

    return None


def filter_known_courses(
    codes: Set[str],
    courses_catalog: Dict[str, Any],
) -> List[str]:
    known_codes: Set[str] = set()

    catalog_codes: Set[str] = set()
    for discipline, courses in courses_catalog.items():
        for code in courses.keys():
            catalog_codes.add(code.upper())

    for code in codes:
        if code.upper() in catalog_codes:
            known_codes.add(code)

    return sorted(known_codes)


def parse_user_string(
    user_message: str,
    manual_courses: Optional[List[str]] = None,
    manual_degree: Optional[str] = None,
    courses_path: Path = COURSES_PATH,
    degrees_path: Path = DEGREES_PATH,
) -> Dict[str, str]:
    
    courses_catalog = load_json(courses_path)
    degrees_catalog = load_json(degrees_path)

    parsed_codes = extract_course_codes(user_message)

    # Allow hard-coded courses to feed to the model in case the string parsing fails
    manual_codes: Set[str] = set()
    if manual_courses:
        for c in manual_courses:
            manual_codes.add(normalize_course_code(c))

    all_codes = parsed_codes | manual_codes

    final_codes = filter_known_courses(all_codes, courses_catalog)

    degree_id = extract_degree_id(user_message, manual_degree) or ""

    info_payload = get_info(
        courses=final_codes,
        degree=degree_id,
        courses_catalog=courses_catalog,
        degrees_catalog=degrees_catalog,
    )

    return {
        "instruction": user_message,
        "input": json.dumps(info_payload, indent=2, ensure_ascii=False),
        "output": "" # blank for inference
    }

# ONLY USED FOR TESTING!
if __name__ == "__main__":
    import sys

    message = "Out of and CS3013, which should I take?"

    example = parse_user_string(
        user_message=message,
        manual_courses=["CS4341", "CS3516"]
    )
    print("Instruction:\n", example["instruction"])
    print("\nInput JSON:\n", example["input"])