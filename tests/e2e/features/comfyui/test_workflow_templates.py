# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from pathlib import Path, PurePath

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WORKFLOW_PATH = (
    Path(__file__).parents[4]
    / "apps"
    / "ComfyUI-vLLM-Omni"
    / "example_workflows"
    / "vLLM-Omni MiniMax H3 Image to Video.json"
)


def test_minimax_h3_i2v_workflow_contract():
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    nodes = workflow["nodes"]
    nodes_by_type = {node["type"]: node for node in nodes}

    assert workflow["version"] == 0.4
    assert sum(node["type"] == "VLLMOmniGenerateVideo" for node in nodes) == 1
    assert {
        "VLLMOmniDiffusionSampling",
        "VLLMOmniGenerateVideo",
        "VLLMOmniMiniMaxH3Params",
        "VLLMOmniRemoteLoRA",
        "SaveVideo",
    } <= nodes_by_type.keys()

    generate = nodes_by_type["VLLMOmniGenerateVideo"]
    generate_inputs = {item["name"]: item for item in generate["inputs"]}
    assert generate_inputs["first_frame"]["link"] is not None
    assert generate_inputs["last_frame"]["link"] is None
    assert generate_inputs["frame"]["link"] is None
    assert generate_inputs["references"]["link"] is None
    assert generate_inputs["lora"]["link"] is None
    assert generate_inputs["fast_h3"]["link"] is None
    assert generate_inputs["duration"]["type"] == "FLOAT"
    assert "num_frames" not in generate_inputs
    assert generate["widgets_values"][:2] == ["http://localhost:8000/v1", "MiniMaxAI/MiniMax-H3"]
    assert generate["widgets_values"][4:7] == [1344, 768, 24]
    assert round(generate["widgets_values"][7] * generate["widgets_values"][6]) == 124
    assert (round(generate["widgets_values"][7] * generate["widgets_values"][6]) - 5) % 17 == 0

    sampling = nodes_by_type["VLLMOmniDiffusionSampling"]
    assert sampling["widgets_values"][:4] == [1, 50, 1, 1]
    h3_params = nodes_by_type["VLLMOmniMiniMaxH3Params"]
    assert h3_params["widgets_values"] == [3, 12]

    image_nodes = [node for node in nodes if node["type"] == "LoadImage"]
    assert {node["widgets_values"][0] for node in image_nodes} == {"first_frame.png", "last_frame.png"}
    assert all(PurePath(node["widgets_values"][0]).name == node["widgets_values"][0] for node in image_nodes)

    lora = nodes_by_type["VLLMOmniRemoteLoRA"]
    assert lora["outputs"][0]["links"] is None
    assert lora["widgets_values"] == [
        "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",
        "h3-turbo-8step",
        1,
        0,
    ]

    save_video = nodes_by_type["SaveVideo"]
    assert save_video["inputs"][0]["link"] is not None
    notes = "\n".join(node["widgets_values"][0] for node in nodes if node["type"] == "MarkdownNote").lower()
    assert all(mode in notes for mode in ("first-frame only", "last-frame only", "first+last-frame"))
    assert all(setting in notes for setting in ("turbo", "num_inference_steps to 9", "flow_shift to 6"))
