from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Experiment 2 evaluation: field-level relaxed macro-F1 over seven clinical-element fields.
# 用法：各模型预测放 results/ 目录，命名为 predictions_<模型名>.json，然后运行本脚本。
# 预测文件格式：JSON对象 {"eval_0000": "模型输出字符串", "eval_0001": "...", ...}
# 缺失的模型文件会被跳过（打警告），不影响其余模型评估。

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"

GOLD_FILE = "data/ground_truth_1463.json"
MODEL_FILES = {
    "YIAN-4B": "results/predictions_YIAN-4B.json",
    "GPT-4.1mini": "results/predictions_GPT-4.1mini.json",
    "GLM4-9B": "results/predictions_GLM4-9B.json",
    "Qwen3-4B": "results/predictions_Qwen3-4B.json",
    "Llama3.1-8B": "results/predictions_Llama3.1-8B.json",
    "Gemma3-4B": "results/predictions_Gemma3-4B.json",
}

FIELDS = (
    "患者信息",
    "主要症状",
    "舌象",
    "脉象",
    "证候",
    "治法",
    "中药",
)

FIELD_ALIASES = {
    "患者信息": ("患者信息", "病人信息", "基本信息", "患者基本信息", "patient_info"),
    "主要症状": ("主要症状", "症状", "临床表现", "主诉", "symptoms"),
    "舌象": ("舌象", "舌诊", "tongue", "tongue_signs"),
    "脉象": ("脉象", "脉诊", "pulse", "pulse_signs"),
    "证候": ("证候", "证型", "辨证", "中医证候", "syndromes", "patterns"),
    "治法": ("治法", "治疗原则", "治疗方法", "treatments"),
    "中药": ("中药", "方药", "药物", "中药列表", "用药", "herbs"),
}

# 仅描述性字段使用宽松匹配；患者信息和中药名规范化后精确匹配。
RELAXED_FIELDS = {"主要症状", "舌象", "脉象", "证候", "治法"}

LIST_SPLIT_RE = re.compile(r"[、，,；;\n\r\t|]+")
EDGE_PUNCT_RE = re.compile(
    r"^[\s:：，,。、；;！？?\-—_\"'“”‘’+|]+|[\s:：，,。、；;！？?\-—_\"'“”‘’+|]+$"
)
DOSE_ONLY_RE = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|克|毫克|千克|毫升|钱|两|枚|片|粒|包|剂|付|帖|味)?$",
    flags=re.IGNORECASE,
)
TRAILING_DOSE_RE = re.compile(
    r"\s*\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|克|毫克|千克|毫升|钱|两|枚|片|粒|包|剂|付|帖|味)?(?:.*)?$",
    flags=re.IGNORECASE,
)
HERB_NAME_KEYS = {"name", "药名", "名称", "中药名", "herb", "herbname", "drugname"}
HERB_METADATA_KEYS = {"dose", "剂量", "unit", "单位", "炮制", "processing", "用法"}


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def metrics(self) -> dict[str, float]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}


def canonical_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"[\s_\-:：]", "", text)


def extract_json_objects(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, str) or not value.strip():
        return []

    text = value.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            parsed, consumed = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                objects.append(parsed)
            cursor = start + consumed
        except json.JSONDecodeError:
            cursor = start + 1
    return objects


def find_value_by_alias(data: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    alias_keys = {canonical_key(alias) for alias in aliases}
    queue: deque[tuple[Mapping[str, Any], int]] = deque([(data, 0)])
    while queue:
        current, depth = queue.popleft()
        for key, value in current.items():
            if canonical_key(key) in alias_keys:
                return value
        if depth >= 3:
            continue
        for value in current.values():
            if isinstance(value, dict):
                queue.append((value, depth + 1))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        queue.append((item, depth + 1))
    return None


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", "", text)
    return EDGE_PUNCT_RE.sub("", text).strip()


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def flatten_general(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [
            item
            for item in (normalize_text(part) for part in LIST_SPLIT_RE.split(value))
            if item
        ]
    if isinstance(value, (int, float, bool)):
        text = normalize_text(value)
        return [text] if text else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(flatten_general(item))
        return result
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(flatten_general(item))
        return result
    return []


def normalize_patient(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return flatten_general(value)
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(normalize_patient(item))
        return result
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                result.extend(normalize_patient(item))
                continue
            text = normalize_text(item)
            if not text:
                continue
            normalized_key = re.sub(r"[\s_\-:：]", "", unicodedata.normalize("NFKC", str(key)).casefold())
            if normalized_key in {"年龄", "age"} and text.isdigit():
                text += "岁"
            result.append(text)
        return result
    text = normalize_text(value)
    return [text] if text else []


def normalize_herb_name(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    text = TRAILING_DOSE_RE.sub("", text)
    return EDGE_PUNCT_RE.sub("", text).strip()


def normalize_herbs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        result: list[str] = []
        for part in flatten_general(value):
            name = normalize_herb_name(part)
            if name and not DOSE_ONLY_RE.fullmatch(name):
                result.append(name)
        return result
    if isinstance(value, list):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and DOSE_ONLY_RE.fullmatch(normalize_text(value[1]))
        ):
            name = normalize_herb_name(value[0])
            return [name] if name else []
        result: list[str] = []
        for item in value:
            result.extend(normalize_herbs(item))
        return result
    if isinstance(value, dict):
        normalized_map = {
            re.sub(r"[\s_\-:：]", "", unicodedata.normalize("NFKC", str(key)).casefold()): item
            for key, item in value.items()
        }
        for name_key in HERB_NAME_KEYS:
            key = re.sub(r"[\s_\-:：]", "", unicodedata.normalize("NFKC", name_key).casefold())
            if key in normalized_map:
                name = normalize_herb_name(normalized_map[key])
                return [name] if name else []

        metadata_keys = {
            re.sub(r"[\s_\-:：]", "", unicodedata.normalize("NFKC", key).casefold())
            for key in HERB_METADATA_KEYS
        }
        non_metadata_keys = [
            key for key in value
            if re.sub(r"[\s_\-:：]", "", unicodedata.normalize("NFKC", str(key)).casefold()) not in metadata_keys
        ]
        if non_metadata_keys and all(not isinstance(value[key], (dict, list)) for key in non_metadata_keys):
            return [name for key in non_metadata_keys if (name := normalize_herb_name(key))]

        result: list[str] = []
        for item in value.values():
            result.extend(normalize_herbs(item))
        return result
    return []


def normalize_field(field: str, value: Any) -> list[str]:
    if field == "患者信息":
        values = normalize_patient(value)
    elif field == "中药":
        values = normalize_herbs(value)
    else:
        values = flatten_general(value)
    return unique_preserving_order(item for item in values if item)


def extract_fields(value: Any) -> tuple[dict[str, list[str]], bool]:
    objects = extract_json_objects(value)
    fields = {field: [] for field in FIELDS}
    for obj in objects:
        for field in FIELDS:
            raw_value = find_value_by_alias(obj, FIELD_ALIASES[field])
            fields[field].extend(normalize_field(field, raw_value))
    return {
        field: unique_preserving_order(values)
        for field, values in fields.items()
    }, bool(objects)


def exact_counts(predicted: Sequence[str], gold: Sequence[str]) -> Counts:
    pred_set = set(predicted)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    return Counts(tp=tp, fp=len(pred_set) - tp, fn=len(gold_set) - tp)


def is_containment_match(left: str, right: str) -> bool:
    """规范化后完全相同或任一字符串包含另一字符串即为候选匹配。"""
    return bool(left and right and (left == right or left in right or right in left))


def maximum_cardinality_matches(edges: list[list[int]]) -> int:
    """在双向包含候选边中求最大数量的一对一匹配。"""
    match_for_gold: dict[int, int] = {}

    def augment(pred_index: int, visited_gold: set[int]) -> bool:
        for gold_index in edges[pred_index]:
            if gold_index in visited_gold:
                continue
            visited_gold.add(gold_index)
            previous_pred = match_for_gold.get(gold_index)
            if previous_pred is None or augment(previous_pred, visited_gold):
                match_for_gold[gold_index] = pred_index
                return True
        return False

    for pred_index in range(len(edges)):
        augment(pred_index, set())
    return len(match_for_gold)


def relaxed_counts(predicted: Sequence[str], gold: Sequence[str], field: str) -> Counts:
    pred_values = unique_preserving_order(predicted)
    gold_values = unique_preserving_order(gold)

    if field not in RELAXED_FIELDS:
        return exact_counts(pred_values, gold_values)

    exact_common = set(pred_values) & set(gold_values)
    pred_remaining = [value for value in pred_values if value not in exact_common]
    gold_remaining = [value for value in gold_values if value not in exact_common]

    edges: list[list[int]] = []
    for pred_value in pred_remaining:
        candidates: list[int] = []
        for gold_index, gold_value in enumerate(gold_remaining):
            if is_containment_match(pred_value, gold_value):
                candidates.append(gold_index)
        edges.append(candidates)

    tp = len(exact_common) + maximum_cardinality_matches(edges)
    return Counts(tp=tp, fp=len(pred_values) - tp, fn=len(gold_values) - tp)


def load_mapping(filename: str) -> dict[str, Any]:
    with (BASE_DIR / filename).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{filename} 的最外层必须是 JSON 对象")
    return data


def aggregate_macro(field_counts: Mapping[str, Counts]) -> dict[str, float]:
    field_metrics = [field_counts[field].metrics() for field in FIELDS]
    return {
        "macro_precision": sum(item["precision"] for item in field_metrics) / len(FIELDS),
        "macro_recall": sum(item["recall"] for item in field_metrics) / len(FIELDS),
        "macro_f1": sum(item["f1"] for item in field_metrics) / len(FIELDS),
    }


def evaluate_model(
    model_name: str,
    predictions: Mapping[str, Any],
    gold_fields: Mapping[str, dict[str, list[str]]],
) -> dict[str, Any]:
    field_counts = {field: Counts() for field in FIELDS}
    for sample_id, sample_gold_fields in gold_fields.items():
        pred_fields, _valid_json = extract_fields(predictions.get(sample_id))
        for field in FIELDS:
            field_counts[field].add(
                relaxed_counts(pred_fields[field], sample_gold_fields[field], field)
            )
    return {"model": model_name, **aggregate_macro(field_counts)}


def write_csv(results: Sequence[Mapping[str, Any]]) -> Path:
    output_path = OUTPUT_DIR / "relaxed_macro_results.csv"
    fieldnames = ["model", "macro_precision", "macro_recall", "macro_f1"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return output_path


def build_report(results: Sequence[Mapping[str, Any]], sample_count: int) -> str:
    lines = [
        "Experiment 2: field-level relaxed macro evaluation results",
        "=" * 78,
        f"评估样本：{sample_count} 条",
        "聚合方法：先在每个字段汇总 TP/FP/FN，再对七个字段等权平均。",
        "患者信息和中药规范化后精确匹配；其余字段采用最大一对一宽松匹配。",
        "宽松规则：规范化后完全相同或双向包含；不使用相似度阈值。",
        "",
        f"{'模型':<22}{'Macro-Precision':>18}{'Macro-Recall':>17}{'Macro-F1':>15}",
        "-" * 78,
    ]
    for result in results:
        lines.append(
            f"{result['model']:<22}"
            f"{result['macro_precision']:>18.2%}"
            f"{result['macro_recall']:>17.2%}"
            f"{result['macro_f1']:>15.2%}"
        )
    return "\n".join(lines)


def main() -> None:
    gold_raw = load_mapping(GOLD_FILE)
    gold_fields: dict[str, dict[str, list[str]]] = {}
    invalid_gold: list[str] = []
    for sample_id, value in gold_raw.items():
        fields, valid = extract_fields(value)
        if not valid:
            invalid_gold.append(sample_id)
        gold_fields[sample_id] = fields
    if invalid_gold:
        raise ValueError(f"金标准有 {len(invalid_gold)} 条无法解析：{', '.join(invalid_gold[:10])}")

    available = {}
    for model_name, filename in MODEL_FILES.items():
        if (BASE_DIR / filename).is_file():
            available[model_name] = filename
        else:
            print(f"[跳过] 未找到 {filename}（模型 {model_name} 不参与本次评估）")

    results = [
        evaluate_model(model_name, load_mapping(filename), gold_fields)
        for model_name, filename in available.items()
    ]

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = write_csv(results)
    report = build_report(results, len(gold_fields))
    report_path = OUTPUT_DIR / "relaxed_macro_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"\nCSV：{csv_path}")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
