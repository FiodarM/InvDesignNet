Neural Network for Inverse Design in Nanophotonics
==================================================
This project demonstates training a neural network for inverse design of nanophotonic gratings.
The project is inspired by the [work](https://doi.org/10.1021/acsphotonics.7b01377) of Liu et al. in ACS Photonics journal.

Generating Data
------------------
* To be able to run DNN training scripts, you should have dataset `dataset.npz` file in the project root folder.
* If you want to use pre-generated dataset, you can download the dataset file from [here](https://drive.google.com/file/d/1D0fJ815a0pgtrE-_lObbhYUnVooeIZIe/view?usp=sharing). **Note**: The file size is ~1 GB.
* In case you wish to generate the dataset from scratch, you can run either [produce_data.py](./produce_data.py) python script or [produce_data.ipynb](./produce_data.py) Jupyter notebook.

Forward Model
-------------
* See [forward_model.ipynb](./forward_model.ipynb) Jupyter notebook for loading, training, and saving forward model
* Saved forward model states can be found in [forward_model](./forward_model/) folder.
  
Inverse Model
--------------
* See [inverse_model.ipynb](./inverse_model.ipynb) Jupyter notebook for loading, training, and saving inverse model
* Saved inverse model states can be found in [inverse_model](./inverse_model/) folder.
