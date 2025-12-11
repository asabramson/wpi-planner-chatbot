import json
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_course_code(code: str) -> str:
    code = code.strip().upper()
    return code.replace(" ", "")

DEGREE_ALIAS_MAP = {
    "BS_CS": "BS Computer Science",
    "BS_DS": "BS Data Science",
    "BS COMPUTER SCIENCE": "BS Computer Science",
    "BS DATA SCIENCE": "BS Data Science",
}

def resolve_degree_name(degree_id: str) -> Optional[str]:
    if not degree_id:
        return None
    key = degree_id.strip().upper()
    return DEGREE_ALIAS_MAP.get(key, degree_id)

def get_info(
    courses: Optional[List[str]],
    degree: Optional[str],
    courses_catalog: Dict[str, Any],
    degrees_catalog: Dict[str, Any],
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    courses_info = []
    if courses:
        for raw_code in courses:
            normalized = normalize_course_code(raw_code)
            discipline = "".join([c for c in normalized if c.isalpha()])  # e.g., "CS"
            course_block = courses_catalog.get(discipline, {}).get(normalized)

            if course_block is None:
                courses_info.append({
                    "code": raw_code,
                    "normalized_code": normalized,
                    "found": False,
                    "error": f"Course {normalized} not found in catalog under discipline {discipline}.",
                })
            else:
                entry = {
                    "code": raw_code,
                    "normalized_code": normalized,
                    "level": course_block.get("level"),
                    "title": course_block.get("title"),
                    "discipline": course_block.get("discipline", discipline),
                    "description": course_block.get("description"),
                    "keywords": course_block.get("keywords", []),
                    "sections": course_block.get("sections", {}),
                }
                courses_info.append(entry)

    if courses_info:
        result["courses_info"] = courses_info

    if degree:
        resolved_name = resolve_degree_name(degree)
        degree_block = degrees_catalog.get(resolved_name)

        if degree_block is None:
            result["degree_info"] = {
                "id": degree,
                "name": resolved_name,
                "found": False,
                "error": f"Degree '{resolved_name}' not found in degrees catalog.",
            }
        else:
            result["degree_info"] = {
                "id": degree,
                "name": resolved_name,
                "found": True,
                "requirements": degree_block,
            }

    return result

BASE_DIR = Path(".")

COURSES_PATH = BASE_DIR / "courses.json"
DEGREES_PATH = BASE_DIR / "degrees.json"
TRAIN_IN_PATH = BASE_DIR / "fine_tuning_corrected.json"
TRAIN_OUT_PATH = BASE_DIR / "fine_tuning_transformed.json"

def parse_get_info_call(call_str: str) -> Dict[str, Any]:
    if not call_str.startswith("get_info"):
        raise ValueError(f"Unsupported input format: {call_str}")

    args_portion = call_str[len("get_info"):].strip()
    expr_str = "dict" + args_portion

    node = ast.parse(expr_str, mode="eval")
    if not isinstance(node, ast.Expression):
        raise ValueError("Could not parse get_info call.")

    call_node = node.body
    if not isinstance(call_node, ast.Call):
        raise ValueError("Expected a function call.")

    kwargs = {}
    for kw in call_node.keywords:
        key = kw.arg
        value = ast.literal_eval(kw.value) if kw.value is not None else None
        kwargs[key] = value

    return kwargs

def main():
    courses_catalog = load_json(COURSES_PATH)
    degrees_catalog = load_json(DEGREES_PATH)

    with TRAIN_IN_PATH.open("r", encoding="utf-8") as f:
        train_data = json.load(f)

    new_data = []

    for example in train_data:
        old_input = example.get("input", "")

        if old_input.startswith("get_info"):
            try:
                params = parse_get_info_call(old_input)
            except Exception as e:
                example["input"] = json.dumps(
                    {"error": f"Failed to parse get_info: {str(e)}",
                     "raw_input": old_input},
                    indent=2
                )
                new_data.append(example)
                continue

            courses_param = params.get("courses") or []
            degree_param = params.get("degree") or ""

            info_payload = get_info(
                courses=courses_param,
                degree=degree_param,
                courses_catalog=courses_catalog,
                degrees_catalog=degrees_catalog,
            )

            example["input"] = json.dumps(info_payload, indent=2, ensure_ascii=False)

        new_data.append(example)

    with TRAIN_OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()