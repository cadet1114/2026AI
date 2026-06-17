from __future__ import annotations

import base64
import copy
import json
import re
import urllib.error
import urllib.request
from typing import Any

from core.demo_engine import calculate_route_cost, clamp


DASHSCOPE_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
ZONE_FIELDS = (
    "sos_signal",
    "building_collapse",
    "smoke",
    "fire",
    "road_damage",
    "human_activity",
    "urgency",
    "congestion",
)


class QwenApiError(RuntimeError):
    """Raised when DashScope returns an API or network error."""


def recognize_disaster_image(
    image_bytes: bytes,
    mime_type: str,
    api_key: str,
    model: str = "qwen-vl-max",
    image_mode: str = "schematic",
) -> tuple[dict[str, Any], str]:
    """Use Qwen-VL to convert a disaster schematic image into structured JSON."""
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = _build_recognition_prompt(image_mode)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 900,
    }
    content = _chat_completion(api_key, payload)
    return _extract_json_object(content), content


def parse_disaster_update_with_qwen(
    update_text: str,
    scenario: dict[str, Any],
    api_key: str,
    model: str = "qwen-max",
    endpoint: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Convert a natural-language disaster update into a safe update_json patch."""
    width = int(scenario.get("map", {}).get("width", 24))
    height = int(scenario.get("map", {}).get("height", 24))
    current_context = {
        "zones": scenario.get("zones", {}),
        "map": {
            "width": width,
            "height": height,
            "targets": scenario.get("map", {}).get("targets", {}),
            "blocked": scenario.get("map", {}).get("blocked", []),
            "fire": scenario.get("map", {}).get("fire", []),
            "congestion": scenario.get("map", {}).get("congestion", []),
            "collapse_cells": scenario.get("map", {}).get("collapse_cells", []),
        },
    }
    prompt = f"""
你是 AI Emergency Commander 的灾情更新解析模块。
请把用户输入的自然语言灾情变化，转换为 update_json。
你只负责提取变化，不要输出救援决策、路线规划或报告。

当前场景摘要：
{json.dumps(current_context, ensure_ascii=False, indent=2)}

用户输入：
{update_text}

只允许输出 JSON，格式如下：
{{
  "updates": [
    {{
      "type": "target_update",
      "target": "A",
      "fields": {{
        "sos_signal": 0.9,
        "fire": 0.8,
        "road_damage": 0.4
      }}
    }},
    {{
      "type": "remove_blocked_cells",
      "cells": [[5, 4]]
    }}
  ]
}}

规则：
- type 只能是 target_update、add_blocked_cells、remove_blocked_cells、add_fire_cells、remove_fire_cells、add_congestion_cells、remove_congestion_cells、add_collapse_cells、remove_collapse_cells。
- target 只能是 A、B、C。
- 如果一句话同时提到多个区域，例如“A区和B区起火，C区火势扩大”，必须为 A、B、C 分别输出 target_update，不要只更新第一个区域。
- fields 只能包含 sos_signal、building_collapse、smoke、fire、road_damage、human_activity、urgency、congestion。
- 所有 fields 数值必须在 0 到 1 之间。
- cells 坐标必须是 [x, y]，x 是 0 到 {width - 1} 的整数，y 是 0 到 {height - 1} 的整数。
- 如果用户描述里没有明确坐标，可以根据 A/B/C 目标点附近给出少量合理格子；不要一次改太多格。
- 如果没有可应用变化，返回 {{"updates": []}}。
- 不要输出 markdown 代码块或解释文字。
""".strip()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You convert Chinese emergency updates into strict JSON patches.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 700,
    }
    content = _chat_completion(
        api_key,
        payload,
        endpoint=endpoint or DASHSCOPE_ENDPOINT,
        provider_name="Local Qwen 7B" if endpoint else "DashScope",
    )
    return _extract_json_object(content), content


def _build_recognition_prompt(image_mode: str) -> str:
    common_shape = """
Return exactly this JSON shape:
{
  "zones": {
    "A": {
      "sos_signal": 0.0,
      "building_collapse": 0.0,
      "smoke": 0.0,
      "fire": 0.0,
      "road_damage": 0.0,
      "human_activity": 0.0,
      "urgency": 0.0,
      "congestion": 0.0
    },
    "B": {},
    "C": {}
  },
  "map": {
    "blocked": [],
    "fire": [],
    "congestion": []
  },
  "notes": "short Chinese note"
}

Rules:
- Every numeric field must be a number from 0 to 1.
- Do not include markdown fences or explanations.
- If exact 24x24 grid coordinates are unclear, keep map lists empty.
""".strip()

    if image_mode == "photo_to_grid":
        return """
You are the vision-to-map module of an AI emergency rescue classroom demo.
Convert the input disaster/scene image into a NEW abstract 24x24 top-down tactical grid map.

Important:
- You are not doing real GIS mapping. You are creating a reasonable classroom-demo grid abstraction from visible evidence.
- Use coordinate format [x, y], where x and y are integers from 0 to 23.
- Bottom-left is [0,0]. Top-right is [23,23].
- Put the rescue base at [2,2] unless the image clearly suggests a better staging area.
- Put the hospital/safe point at [21,21] unless the image clearly suggests a better safe area.
- Create three target zones:
  A = strongest trapped-person / collapse / rescue-priority evidence
  B = relatively safer or more accessible area
  C = highest fire/smoke/road-uncertainty risk
- Infer blocked/fire/congestion cells from visible obstacles, flames, smoke, traffic, debris, and damaged roads.
- If the image is ambiguous, still create a plausible 24x24 map. Do not leave all map lists empty.

Return JSON only, exactly this shape:
{
  "zones": {
    "A": {
      "sos_signal": 0.0,
      "building_collapse": 0.0,
      "smoke": 0.0,
      "fire": 0.0,
      "road_damage": 0.0,
      "human_activity": 0.0,
      "urgency": 0.0,
      "congestion": 0.0
    },
    "B": {
      "sos_signal": 0.0,
      "building_collapse": 0.0,
      "smoke": 0.0,
      "fire": 0.0,
      "road_damage": 0.0,
      "human_activity": 0.0,
      "urgency": 0.0,
      "congestion": 0.0
    },
    "C": {
      "sos_signal": 0.0,
      "building_collapse": 0.0,
      "smoke": 0.0,
      "fire": 0.0,
      "road_damage": 0.0,
      "human_activity": 0.0,
      "urgency": 0.0,
      "congestion": 0.0
    }
  },
  "map": {
    "width": 24,
    "height": 24,
    "base": [2, 2],
    "hospital": [21, 21],
    "targets": {
      "A": [17, 6],
      "B": [8, 17],
      "C": [19, 14]
    },
    "roads": [[2, 2], [3, 2], [4, 2]],
    "buildings": [[16, 4]],
    "water": [],
    "park": [],
    "blocked": [[12, 10]],
    "fire": [[19, 14]],
    "congestion": [[6, 3]],
    "collapse_cells": [[16, 5]]
  },
  "notes": "short Chinese note explaining how the 24x24 map was abstracted from the image"
}

Rules:
- Every numeric disaster field must be from 0 to 1.
- blocked/fire/congestion/collapse_cells must contain grid coordinates only.
- Avoid putting blocked cells on base, hospital, or A/B/C target cells.
- Do not include markdown fences or explanations.
""".strip()

    return """
You are the vision-to-map module of an AI emergency rescue classroom demo.
Read the labeled disaster grid input map and convert it into the system 24x24 scenario JSON.

Important:
- Use coordinate format [x, y], where x and y are integers from 0 to 23.
- Bottom-left is [0,0]. Top-right is [23,23].
- Read the marked cells exactly when coordinates are visible.
- Detect S as rescue base, H as hospital, A/B/C as target zones.
- Detect BLOCK as blocked road cells, COLL as collapse cells, FIRE as fire risk cells, JAM as congestion cells.
- If the image has a coordinate list using another grid size, scale or remap it into the 24x24 output grid.

Return JSON only, exactly this shape:
{
  "zones": {
    "A": {
      "sos_signal": 0.0,
      "building_collapse": 0.0,
      "smoke": 0.0,
      "fire": 0.0,
      "road_damage": 0.0,
      "human_activity": 0.0,
      "urgency": 0.0,
      "congestion": 0.0
    },
    "B": {
      "sos_signal": 0.0,
      "building_collapse": 0.0,
      "smoke": 0.0,
      "fire": 0.0,
      "road_damage": 0.0,
      "human_activity": 0.0,
      "urgency": 0.0,
      "congestion": 0.0
    },
    "C": {
      "sos_signal": 0.0,
      "building_collapse": 0.0,
      "smoke": 0.0,
      "fire": 0.0,
      "road_damage": 0.0,
      "human_activity": 0.0,
      "urgency": 0.0,
      "congestion": 0.0
    }
  },
  "map": {
    "width": 24,
    "height": 24,
    "base": [2, 2],
    "hospital": [21, 21],
    "targets": {
      "A": [17, 6],
      "B": [8, 17],
      "C": [19, 14]
    },
    "roads": [],
    "buildings": [],
    "water": [],
    "park": [],
    "blocked": [],
    "fire": [],
    "congestion": [],
    "collapse_cells": []
  },
  "notes": "short Chinese note"
}

Rules:
- Every numeric disaster field must be from 0 to 1.
- blocked/fire/congestion/collapse_cells must contain grid coordinates only.
- Do not include markdown fences or explanations.
""".strip()


def generate_qwen_report(
    scenario: dict[str, Any],
    zone_scores: dict[str, dict[str, float]],
    assignments: dict[str, str],
    routes: dict[str, list[list[int]]],
    api_key: str,
    model: str = "qwen-max",
    endpoint: str | None = None,
    route_details: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Generate a Chinese rescue report from deterministic algorithm outputs."""
    display_assignments = {
        unit: ("无人机侦查任务" if unit == "Drone-1" else f"{target}区救援")
        for unit, target in assignments.items()
    }
    data = {
        "zone_scores": zone_scores,
        "assignments": display_assignments,
        "route_lengths": {
            unit: max(len(route) - 1, 0) for unit, route in routes.items()
        },
        "route_costs": {
            unit: _route_cost_for_report(scenario, unit, route, route_details)
            for unit, route in routes.items()
        },
        "collapsed": bool(scenario.get("map", {}).get("collapsed")),
    }
    prompt = (
        "请根据下面的算法结果生成一份课堂演示用中文救援报告。"
        "要求：分成“当前判断 / 任务与路线 / 重规划说明”三段，"
        "每段 1-2 句；不要夸大真实救援能力；说明先救谁、为什么、派车/无人机原因、"
        "路径规划和动态重规划结果。无人机任务统一表述为“无人机侦查”，"
        "不要写成直接前往某个具体区域侦查。控制在 180 字以内。\n\n"
        + json.dumps(data, ensure_ascii=False, indent=2)
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise emergency-command report writer.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 320,
    }
    return _chat_completion(
        api_key,
        payload,
        endpoint=endpoint or DASHSCOPE_ENDPOINT,
        provider_name="Local Qwen 7B" if endpoint else "DashScope",
    ).strip()


def _route_cost_for_report(
    scenario: dict[str, Any],
    unit: str,
    route: list[list[int]],
    route_details: dict[str, dict[str, Any]] | None,
) -> float:
    detail = (route_details or {}).get(unit, {})
    if isinstance(detail.get("total_cost"), (int, float)):
        return round(float(detail["total_cost"]), 2)
    return calculate_route_cost(scenario, unit, route)


def merge_recognition_into_scenario(
    scenario: dict[str, Any],
    recognition: dict[str, Any],
    update_map: bool = False,
) -> dict[str, Any]:
    """Merge Qwen-VL JSON into the existing demo scenario safely."""
    updated = copy.deepcopy(scenario)
    zones = recognition.get("zones", recognition)

    for zone_name in ("A", "B", "C"):
        if not isinstance(zones.get(zone_name), dict):
            continue
        updated["zones"].setdefault(zone_name, {})
        for field in ZONE_FIELDS:
            value = zones[zone_name].get(field)
            if isinstance(value, (int, float)):
                updated["zones"][zone_name][field] = round(clamp(float(value)), 3)

    if update_map and isinstance(recognition.get("map"), dict):
        map_data = recognition["map"]
        width = int(updated["map"]["width"])
        height = int(updated["map"]["height"])
        for point_key in ("base", "hospital"):
            point = _normalize_point(map_data.get(point_key), width, height)
            if point:
                updated["map"][point_key] = point

        if isinstance(map_data.get("targets"), dict):
            updated["map"].setdefault("targets", {})
            for zone_name in ("A", "B", "C"):
                target = _normalize_point(map_data["targets"].get(zone_name), width, height)
                if target:
                    updated["map"]["targets"][zone_name] = target

        for key in ("roads", "buildings", "water", "park", "blocked", "fire", "congestion"):
            cells = _normalize_cells(map_data.get(key), width, height)
            if cells:
                updated["map"][key] = cells
        collapse_cells = _normalize_cells(map_data.get("collapse_cells"), width, height)
        if collapse_cells:
            updated["map"]["collapse_cells"] = collapse_cells

    updated["qwen_notes"] = recognition.get("notes", "")
    return updated


def _chat_completion(
    api_key: str,
    payload: dict[str, Any],
    endpoint: str = DASHSCOPE_ENDPOINT,
    provider_name: str = "DashScope",
) -> str:
    if endpoint == DASHSCOPE_ENDPOINT and not api_key:
        raise QwenApiError("DASHSCOPE_API_KEY is missing.")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise QwenApiError(f"{provider_name} HTTP {exc.code}: {body[:600]}") from exc
    except Exception as exc:
        raise QwenApiError(f"{provider_name}: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise QwenApiError(f"Unexpected {provider_name} response: {data}") from exc


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S | re.I)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise QwenApiError(f"Qwen-VL did not return JSON: {text[:300]}")

    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise QwenApiError(f"Invalid JSON from Qwen-VL: {text[:500]}") from exc


def _normalize_cells(value: Any, width: int, height: int) -> list[list[int]]:
    if not isinstance(value, list):
        return []

    cells: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for item in value:
        if (
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(part, (int, float)) for part in item)
        ):
            x, y = int(item[0]), int(item[1])
            if 0 <= x < width and 0 <= y < height and (x, y) not in seen:
                seen.add((x, y))
                cells.append([x, y])
    return cells


def _normalize_point(value: Any, width: int, height: int) -> list[int]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(part, (int, float)) for part in value)
    ):
        x, y = int(value[0]), int(value[1])
        if 0 <= x < width and 0 <= y < height:
            return [x, y]
    return []
