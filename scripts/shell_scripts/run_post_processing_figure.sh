#!/bin/bash -l

# Request 1 core per task
#$ -pe omp 1
# Set runtime limit
#$ -l h_rt=12:00:00
# Join output and error streams
#$ -j y
# Give the job a name
#$ -N post_process_figure

module load python3/3.10.12
export PYTHONPATH=$PYTHONPATH:/projectnb/aclab/jiujiaz/adapt-tb
source /projectnb/aclab/jiujiaz/adapt-tb/venv/bin/activate

python3 scripts/post_processing/generate_figure.py
