# GSA_algorithm

This repository includes the functions to perform the Generalised Smoothing Algorithm (GSA), first developed by Ramsay et al. [2007]:

- `main.py` is an example of the main script to perform GSA using the example function defined in `fun_examples` and the data example `database.txt`.
- `spline.py` defines the methods to work with splines.
- `GSA_fun.py` contains all the methods needed to perform a GSA on data.

# GSPINN_algorithm

This repository also includes the functions to perform the Generalised Smoothing Physics Informed Neural Network:
in the  folder, different scripts to run to test the different parts of the algorithm.

## User-specific model
To use the GSA for a specific application, employing a specific model, we invite the user to create a new file within the folder `fun_example` and define the model.
An example is provided for the Generalised Lotka-Volterra (GLV) model in `GLV_functions.py`.

## User-specific data
To use the GSA with a model already available in the repo, but on different data, we suggest to use the format `txt` and upload the database in the folder `data_examples`.
An example of database for the GLV model is provided as `database.txt`.


