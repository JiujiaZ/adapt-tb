# adapt-tb
This is code repository for adapt-tb.

## Getting Started
0. Make sure data/input is given (currently hidden due to data sharing restriction)
1. Clone the repository:
   ```bash
   git clone https://github.com/JiujiaZ/adapt-tb.git
   ```
2. Navigate to the repository directory:
   ```bash
   cd adapt-tb
   ```
## Requirements
- Python 3.10.12 or upto 3.11
- All dependencies listed in `requirements.txt`

## Set up Environment

Before running any tasks, you need to set the `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:<path>/adapt-tb
````
## Usage:
We use K, r, d to denote # units, latter two are associated model parameters.

Current version supports:
- K: 1, 2, 3, 4
- r: 1, 2, 3, 4, 5
- d: 0.33, 0.43
                        
1. Run model parameter fitting:
  ``` bash
  python main/run_fit_model.py --r <r> --d <d>
  ```
This saves fitted model parameters as data/parameters/parameters_r<r>_d<d*100>.pkl

2. Run model validation:
  ``` bash
  python main/run_validate_model.py --r <r> --d <d>
  ```
This saves raw output from model validation as data/output/validation/simulated_data_r<r>_d<d*100>.npz

A plot is also generated using raw data being saved to results/validation/validation_r<r>_d<d*100>.png

3. Run screening yield simulation:
  ``` bash
  python main/run_simulate_model.py --K <K> --r <r> --d <d>
  ```
This saves raw output from simulation as data/output/simulation/simulated_data_K<K>_r<r>_d<d*100>.npz

A plot is also generated using raw data being saved to results/simulation/simulation_K<K>_r<r>_d<d*100>.png

### Runtime Estimate
- Tasks typically take less than 20 minutes, 20 minutes, or up to 4 hours for each run, depending on the configuration.

## License
This repository is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) – see the [LICENSE](https://creativecommons.org/licenses/by/4.0/) file for details.
