# WindCal

Python implementation of Recommended Practice `AIAA R-091A-2020`
*Calibration and Use of Internal Strain-Gage Balances with Application to Wind
Tunnel Testing*.

## Objective
WindCal is a flexible, object-oriented Python library designed to handle the calibration and data 
reduction of internal strain-gage balances. It is built to seamlessly integrate into both high-speed, 
live Data Acquisition (DAQ) systems and legacy post-processing workflow.

Key features include:
- **Interchangeable Math Models:** Hot-swap between simple 6x6 linear matrix solutions and complex 
iterative polynomial solvers.
- **Balance Geometry Agnostic:** Built-in coordinate resolution to dynamically handle 3F/3M, force,
or moment balance configurations.
- **Auditable Artifacts:** Calibration matrices and their associated metadata are serialized as 
versionable artifacts, ensuring a single "source of truth" throughout the lab.

## Example Usage

See the `docs/examples/` directory for sample usage.
