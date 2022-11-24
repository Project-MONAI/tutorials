#! /bin/bash

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python ./luna16_post_combine_cross_fold_results.py \
	-i ./result/result_luna16_fold0.json \
	./result/result_luna16_fold1.json \
	./result/result_luna16_fold2.json \
	./result/result_luna16_fold3.json \
	./result/result_luna16_fold4.json \
	./result/result_luna16_fold5.json \
	./result/result_luna16_fold6.json \
	./result/result_luna16_fold7.json \
	./result/result_luna16_fold8.json \
	./result/result_luna16_fold9.json \
	-o ./result/result_luna16_all.csv

mkdir -p ./result/eval_luna16_scores
python2 ./evaluation_luna16/noduleCADEvaluationLUNA16.py \
	./evaluation_luna16/annotations/annotations.csv  \
	./evaluation_luna16/annotations/annotations_excluded.csv \
	./evaluation_luna16/annotations/seriesuids.csv \
	./result/result_luna16_all.csv \
	./result/eval_luna16_scores
