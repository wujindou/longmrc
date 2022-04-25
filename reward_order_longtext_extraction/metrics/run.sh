#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
# 以下是样例，你可以自定义修改
python metrics.py \
    --predict_result_file_dir=$PREDICT_RESULT_FILE_DIR \
    --groudtruth_file_dir=$GROUNDTRUTH_FILE_DIR \
    --result_json_file=$RESULT_JSON_FILE \
    --result_detail_file=$RESULT_DETAIL_FILE
