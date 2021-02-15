Neural Network for Inverse Design in Nanophotonics
==================================================
This project demonstates training a neural network for inverse design of nanophotonic gratings.
The project is inspired by the [work](https://doi.org/10.1021/acsphotonics.7b01377) of Liu et al. in ACS Photonics journal.

Generating Data
------------------
* The pregenerated dataset is stored in dataset.npz file.
* In case you wish to append samples to the dataset or regenerate the dataset yourself, you can run either [produce_data.py](./produce_data.py) python script or [produce_data.ipynb](./produce_data.py) Jupyter notebook.

Forward Model
-------------
* See [forward_model.ipynb](./forward_model.ipynb) Jupyter notebook for loading, training, and saving forward model
* Saved forward model states can be found in [forward_model](./forward_model/) folder.
  
Inverse Model
--------------
* See [inverse_model.ipynb](./inverse_model.ipynb) Jupyter notebook for loading, training, and saving forward model
* Saved inverse model states can be found in [inverse_model](./inverse_model/) folder.
