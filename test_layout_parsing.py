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
# import os
# from glob import glob
# from paddlex import create_pipeline
from paddlex.inference.pipelines.layout_parsing.utils import direct_test
from eval_layout_parsing import main
import json

num = 70
page_idx = 1
test = 0
if test == 0:
    direct_test(
        f"xxx/DocBench-100/{num}/input_{num}.json",
        f"xxx/DocBench-100/{num}/out_{num}.json",
        min_gap_x=1000,
        min_gap_y=6000,
        is_only_x=True,
    )
    main(debug=True, start_idx=0, end_idx=1, num=num)
else:
    direct_test(
        f"xxx/DocBench-100/{num}/input_{num}.json",
        f"xxx/DocBench-100/{num}/out_{num}.json",
        min_gap_x=1000,
        min_gap_y=6000,
        is_only_x=True,
        start_page_id=page_idx,
        end_page_id=page_idx + 1,
    )
    main(
        debug=True,
        start_idx=0,
        end_idx=1,
        page_start_idx=page_idx,
        page_end_idx=page_idx + 1,
        num=num,
    )

# direct_test(
#     f"xxx/input_{num}.json",
#     f"xxx/out_{num}.json",
# )

# keys = ["1andmore_column", "double_column", "three_column", "single_column",]
# index = 3
# key = keys[index]
# page_idx = 3
# test = 0

# if test == 0:
#     direct_test(
#         f"xxx/input_jsons/input_{key}.json",
#         f"xxx/input_jsons/output_{key}.json",
#         min_gap_x=1,
#         min_gap_y=6,
#         is_only_x=True,
#     )
#     main(debug=True, start_idx=index, end_idx=index + 1)
# else:
#     direct_test(
#         f"xxx/input_jsons/input_{key}.json",
#         f"xxx/input_jsons/output_{key}.json",
#         min_gap_x=1,
#         min_gap_y=6,
#         is_only_x=True,
#         start_page_id=page_idx,
#         end_page_id=page_idx + 1,
#     )
#     main(
#         debug=True,
#         start_idx=index,
#         end_idx=index + 1,
#         page_start_idx=page_idx,
#         page_end_idx=page_idx + 1,
#     )

