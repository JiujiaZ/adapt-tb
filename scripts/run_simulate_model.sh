#!/bin/bash -l

# Request array job with multiple tasks (one task per combination of K, r and d)
#$ -t 1-100
# Request 1 core per task
#$ -pe omp 1
# Set runtime limit
#$ -l h_rt=12:00:00
# Join output and error streams
#$ -j y
# Give the job a name
#$ -N model_simulation_job


module load python3/3.10.12
export PYTHONPATH=$PYTHONPATH:/projectnb/aclab/jiujiaz/adapt-tb
source /projectnb/aclab/jiujiaz/adapt-tb/venv/bin/activate

# Array of r,d,K values
r_values=(1 2 3 4 5)
d_values=(0.33 0.43 0.53 0.63 0.73)
K_values=(1 2 3 4)

# Calculate total number of combinations
total_combinations=$((${#K_values[@]} * ${#r_values[@]} * ${#d_values[@]}))

# Ensure SGE_TASK_ID is within bounds (1 <= SGE_TASK_ID <= total_combinations)
if [ "$SGE_TASK_ID" -gt "$total_combinations" ]; then
  echo "Error: SGE_TASK_ID ($SGE_TASK_ID) is greater than the number of available combinations ($total_combinations). Exiting."
  exit 1
fi

# Determine r d K correspond to this SGE_TASK_ID
d_index=$(( (SGE_TASK_ID - 1) / (${#K_values[@]} * ${#r_values[@]}) ))
r_index=$(( (SGE_TASK_ID - 1) % (${#K_values[@]} * ${#r_values[@]}) / ${#K_values[@]} ))
K_index=$(( (SGE_TASK_ID - 1) % ${#K_values[@]} ))


K=${K_values[$K_index]}
r=${r_values[$r_index]}
d=${d_values[$d_index]}

echo "Running model for K=$K, r=$r, d=$d (SGE_TASK_ID=$SGE_TASK_ID)"

# Run the Python script with the selected K, r and d
python3 main/run_simulate_model.py --K $K --r $r --d $d

# Check if the Python script ran successfully
if [ $? -ne 0 ]; then
  echo "Model failed for K=$K, r=$r, d=$d. Exiting."
  exit 1
fi

echo "Model completed successfully for K=$K, r=$r, d=$d"

