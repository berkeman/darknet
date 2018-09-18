#!/bin/bash
for x in alexnet darknet53 densenet201 resnet152 vgg-16 vgg-conv yolov1 yolov2 yolov3;\
do \
  python3 wextract.py cfg/${x}.cfg | python3 averager.py > out/${x}.out; \
done	     

for x in cfg/*.cfg; do
  echo ${x} && (python3 ./wextract.py ${x}  | awk 'BEGIN{s=0}{s+=$9;print $9,s}' | tail -1);
done | paste - - > sum_weights.txt

for x in cfg/*.cfg; do
  echo ${x} && (python3 ./wextract.py ${x}  | awk 'BEGIN{s=0}{s+=$8;print $8,s}' | tail -1);
done | paste - - > sum_macs.txt





	 