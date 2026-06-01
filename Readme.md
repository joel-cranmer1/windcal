# WindCal

Python implementation of Recommended Practice
`AIAA R-091A-2020` *Calibration and Use of Internal Strain-Gage Balances with Application to Wind
Tunnel Testing*.

## Objective
WindCal is a flexible, object-oriented Python library designed to handle the calibration and data 
reduction of internal strain-gage balances. It is built to seamlessly integrate into both high-speed, 
live Data Acquisition (DAQ) systems and legacy post-processing Graphical User Interfaces (GUIs). 

Key features include:
- **Interchangeable Math Models:** Hot-swap between simple 6x6 linear matrix solutions and complex 
iterative polynomial solvers.
- **Balance Geometry Agnostic:** Built-in coordinate resolution to dynamically handle 3F/3M and 
5F/1M balance configurations.
- **Auditable Artifacts:** Calibration matrices and their associated metadata are serialized as 
versionable artifacts, ensuring a single "source of truth" throughout the lab.

## Example Usage

### 1. Generating a Calibration
Create a balance calibration artifact from known loads. This is an auditable self-contained file.

```python
from windcal.core.io import CalibrationDataSet
from windcal.calibration.models import Linear6x6Model
from windcal.calibration.calibrator import Calibrator

# Load known applied loads and voltage responses
dataset = CalibrationDataSet.from_csv("raw_cal_data.csv")

# Select a math strategy and fit the data
calibrator = Calibrator(math_model=Linear6x6Model())
calibration_artifact = calibrator.generate_calibration(dataset, author="J. Cranmer")

# Save as a versioned JSON artifact
calibration_artifact.save("balance_sn123_v1.json")
```

### 2. Live DAQ Data Reduction
Use the fast `process_point` method inside your wind tunnel's live control loop.

```python
from windcal.core.artifact import BalanceCalibration
from windcal.reduction.reducer import DataReducer

# Load the approved calibration artifact
approved_cal = BalanceCalibration.load("balance_sn123_v1.json")
reducer = DataReducer(calibration=approved_cal)

while daq_is_running:
    # Get a 1D numpy array of 6 voltages from hardware
    raw_voltages = daq_hardware.read_channels() 
    
    # Fast, single-point reduction to physical loads
    live_loads = reducer.process_point(raw_voltages) 
```

### 3. Batch Post-Processing
Process entire legacy run files efficiently using vectorized operations.

```python
import pandas as pd
from windcal.core.artifact import BalanceCalibration
from windcal.reduction.reducer import DataReducer

approved_cal = BalanceCalibration.load("balance_sn123_v1.json")
reducer = DataReducer(calibration=approved_cal)

# Load batch voltages (e.g., from a CSV DataFrame)
voltage_df = pd.read_csv("run_042_voltages.csv")

# Vectorized reduction of the entire DataFrame
loads_df = reducer.process_batch(voltage_df)
loads_df.to_csv("run_042_loads.csv")
```
