
# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddlex.inference.pipelines.layout_parsing.utils import direct_test
from eval_layout_parsing import main
import json

min_gap_x = 1
min_gap_y = 6
is_only_x = False
keys = ["1andmore_column", "single_column", "double_column", "three_column"]

all_results = [] 
best_results = []  
error_records = [] 

try:
    current_mean_bleu = 0
    print(
        f"Testing config: min_gap_x={min_gap_x}, min_gap_y={min_gap_y}, is_only_x={is_only_x}"
    )
    time_total = 0.0
    count_t = 0
    for key in keys:
        time_t, count_t = direct_test(
            f"xxx/input_jsons/input_{key}.json",
            f"xxx/input_jsons/output_{key}.json",
            min_gap_x=min_gap_x,
            min_gap_y=min_gap_y,
            is_only_x=is_only_x,
        )
        time_total += time_t
        count_t += count_t
    print(count_t / time_total, "fps")
    bleu_score, ard, tau, edit_dist = main()
    current_mean_bleu = bleu_score / 4
    current_results = {
        "config": {
            "min_gap_x": min_gap_x,
            "min_gap_y": min_gap_y,
            "is_only_x": is_only_x,
        },
        "results": {
            "mean_bleu": current_mean_bleu,
            "bleu_score": bleu_score,
            "ard": ard,
            "tau": tau,
            "edit_dist": edit_dist,
        },
    }
    all_results.append(current_results)
    
    best_results.append(current_results)
    best_results.sort(key=lambda x: x["results"]["mean_bleu"], reverse=True)
    best_results = best_results[:10]
    print(f"Current mean BLEU: {current_mean_bleu}")
except Exception as e:
    error_records.append(
        {
            "config": {
                "min_gap_x": min_gap_x,
                "min_gap_y": min_gap_y,
                "is_only_x": is_only_x,
            },
            "error": str(e),
        }
    )
    print(
        f"Error occurred with config (min_gap_x={min_gap_x}, min_gap_y={min_gap_y}, is_only_x={is_only_x}): {e}"
    )


for idx, result in enumerate(best_results):
    print(f"Rank {idx + 1}:")
    print(
        f"  Config: min_gap_x={result['config']['min_gap_x']}, min_gap_y={result['config']['min_gap_y']}, is_only_x={result['config']['is_only_x']}"
    )
    print(f"  Mean BLEU: {result['results']['mean_bleu']}")
    print(f"  BLEU Score: {result['results']['bleu_score']}")
    print(f"  ARD: {result['results']['ard']}")
    print(f"  Tau: {result['results']['tau']}")
    print(f"  Edit Distance: {result['results']['edit_dist']}")
