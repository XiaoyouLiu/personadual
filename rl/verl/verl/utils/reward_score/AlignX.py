# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import json


def format_reward(predict_str: str) -> float:
    pattern = re.compile(
    r'.*?\n\n<answer>\s*\{"answer":.*?\}\s*</answer>$',
    re.DOTALL
)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    matches = re.findall(r'\{[\s\S]*?\}', predict_str)
    if matches:
        try:
            ans = json.loads(matches[-1]).get('answer', '')
        except json.decoder.JSONDecodeError:
            ans = None
    else:
        ans = None
    return 1.0 if ans == ground_truth else 0.0


def compute_score(predict_str: str, ground_truth: str, format_score: float = 0.1) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth) + format_score * format_reward(
        predict_str
    )
