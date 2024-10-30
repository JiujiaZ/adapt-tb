#!/bin/bash -l

# Request array (one task per combination of r and d)
#$ -t 1-15
# Request 1 core per task
#$ -pe omp 1
# Set runtime limit
#$ -l h_rt=12:00:00
# Join output and error streams
#$ -j y
# Give the job a name
#$ -N model_validation_job


module load python3/3.10.12
export PYTHONPATH=$PYTHONPATH:/projectnb/aclab/jiujiaz/adapt-tb
source /projectnb/aclab/jiujiaz/adapt-tb/venv/bin/activate

# Array of r values
r_values=(1 2 3 4 5)

# Array of d values
d_values=(0.23 0.33 0.43)

# Calculate total number of combinations
total_combinations=$((${#r_values[@]} * ${#d_values[@]}))

# Ensure SGE_TASK_ID is within bounds (1 <= SGE_TASK_ID <= total_combinations)
if [ "$SGE_TASK_ID" -gt "$total_combinations" ]; then
  echo "Error: SGE_TASK_ID ($SGE_TASK_ID) is greater than the number of available combinations ($total_combinations). Exiting."
  exit 1
fi

# Determine which r and d values correspond to this SGE_TASK_ID
r_index=$(( (SGE_TASK_ID - 1) % ${#r_values[@]} ))
d_index=$(( (SGE_TASK_ID - 1) / ${#r_values[@]} ))

r=${r_values[$r_index]}
d=${d_values[$d_index]}

echo "Running model for r=$r, d=$d (SGE_TASK_ID=$SGE_TASK_ID)"

# Run the Python script with the selected r and d
python3 main/run_validate_model.py --r $r --d $d

# Check if the Python script ran successfully
if [ $? -ne 0 ]; then
  echo "Model failed for r=$r, d=$d. Exiting."
  exit 1
fi

echo "Model completed successfully for r=$r, d=$d"

