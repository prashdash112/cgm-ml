#!/bin/sh

echo "Running training..."
start=`date +%s`
python3 train_pointnet_generator.py
end=`date +%s`
runtime=$((end-start))
echo "Duration:" $runtime
