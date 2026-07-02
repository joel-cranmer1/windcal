# WindCal

> Python implementation of Recommended Practice `AIAA R-091A-2020`
> *Calibration and Use of Internal Strain-Gage Balances with Application to Wind Tunnel Testing*.

> [!warning] Work in Progress:
> WindCal is currently a work in progress. Expect instability with API.

[![numpy-badge][numpy-img]][numpy-url]
[![pandas-badge][pandas-img]][pandas-url]
[![python-version-badge][python-img]][python-url]

WindCal is a flexible, object-oriented Python library designed for the calibration and data reduction of internal strain-gage balances used in wind tunnel testing. It supports both real-time data acquisition (DAQ) integration and legacy post-processing workflows, making it suitable for modern and existing lab environments.

The library emphasizes modularity, traceability, and mathematical flexibility. Calibration artifacts and metadata are treated as versionable, auditable objects, ensuring reproducibility and consistency across experiments and teams.


## Installation

```sh
python -m pip install git+https://gitlab.cicrange.net/all/dfan/apps/windcal.git
```

## Usage example

See the [`docs/examples/`](docs/examples/) directory for sample usage.

Key features include:
- **Interchangeable Math Models:** Hot-swap between simple 6x6 linear matrix solutions and complex iterative polynomial solvers.
- **Balance Geometry Agnostic:** Built-in coordinate resolution to dynamically handle 3F/3M, force, or moment balance configurations.
- **Auditable Artifacts:** Calibration matrices and their associated metadata are serialized as versionable artifacts, ensuring a single "source of truth" throughout the lab.


## Development setup

To set up a development environment:

```sh
python -m pip install -e .[dev]
```

Run ruff code linter:

```sh
ruff check .
```

Run tests:

```sh
python -m unittest discover
```

## Roadmap

- [x] Implement Linear Model
- [ ] Implement Linear with absolute value Model
- [ ] Implement polynomial model
- [ ] Implement aerodynamic coefficent reduction procedure
    - [ ] Collect Geometry Data
    - [ ] Ingest flow conditions (temperature, pressure, orientation, etc)

## Release History

- 0.1.0
    - Initial Work in progress.

## Meta

USAFA/DFAN Joel Cranmer - joel.cranmer@afacademy.af.edu

Distributed under the XYZ license. See ``LICENSE`` for more information.

@joel.cranmer

## Contributing

1. Fork it (https://gitlab.cicrange.net/all/dfan/apps/windcal/fork)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[numpy-img]: https://img.shields.io/badge/using-numpy-blue?logo=numpy&logoColor=white
[numpy-url]: https://pypi.org/project/numpy/
[pandas-img]: https://img.shields.io/badge/using-pandas-blue?logo=python&logoColor=white
[pandas-url]: https://pypi.org/project/pandas/
[python-img]: https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white
[python-url]: https://www.python.org/downloads/
