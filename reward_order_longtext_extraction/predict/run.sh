#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
#cd $basepath/../../
#source env.sh
#cd $basepath/../
#source setting.conf
cd $basepath


# 以下是样例，你可以自定义修改
python predict.py \
    --predict_file_dir=/search/odin/jdwu/lic_2022/test/ \
    --experiment_path=/search/odin/jdwu/lic_2022/reward_order_longtext_extraction/train/output/ \
    --predict_result_file_dir=/search/odin/jdwu/lic_2022/result \
    --pretain_model="../../model/roberta_zh/"
