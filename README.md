# satellite-satellite-near-coincidence-solver

Code for identifying and evaluating joint domains between two satellites based on a radius of convergence and a maximum temporal displacement between satellites at their point of crossing.

This will utilise two line element (TLE) sets or OMM files (e.g. as provided by [Celestrak](https://celestrak.org/)) to describe the orbits of satellites. To increase the accuracy of the results, sets of TLE orbital parameters can be provided that cover different epochs within the time frame of interest. These can be obtained through [special data requests](https://celestrak.org/NORAD/archives/request.php) to Celestrak.

## Installation

To install the package, locally called `ssncs`, you should install the dependencies for the package by using the command

```
mamba env create --file environment.yml
```

You can then install `ssncs` by typing `pip install -e .` from within the root of this directory, which will create an editable install of the `ssncs` package within the newly created `ssncs` conda environment.
