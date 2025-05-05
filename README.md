Here we explain how to reproduce the synthetic data test described in Section 3 of the paper.

You need `GMT` and the following Python libraries: `GeoPandas`, `japanmap`, `Shapely`, `NumPy`, `Matplotlib`, `ObsPy`, `SciPy`, `netCDF4`, `seaborn`, `DeepXDE`, `Tensorflow`, `PyGMT`.
Also, please install `JAGURS`. See the `JAGURS` manual for details.

# Preparation
- Go to `share`.
- Run `prepare_bathy.sh`. (Download bathymetry data of eastern Japan, cut & resample.)
- Run `prepare_cols.ipynb`. (Generate collocation points and save them in a file.) Note that this method cannot be used outside Japan.


# Synthesize data
- Compile `JAGURS` (as serial version, i.e., without MPI) and create an executable file. In `Makefile`, specify `EXEC=jagurs_test`, `MPI=OFF`, and nothing is specified for `OUTPUT`. Please put the executable file (`jagurs_test`) in the same directory as this readme.
- Go to `data_Tohoku` and run the simulation (see `run_JAGURS1.sh`).
- Format the data with `process.ipynb`.

If you use actual data (S-net, N-net, etc.), please do the similar process as `process.ipynb` by yourself.


# PINN training
`DeepXDE` supports several backends. Here we use `TensorFlow (v1)`, but you can also choose `TensorFlow (v2)` or `PyTorch`. See the `DeepXDE` documentation for details.
- Run `train.py`. (see `run_PINN.sh`).
- Run `postprocess1.ipynb` to create the figures of the wavefield estimation.
- Run `run_JAGURS2.sh` at `result_Tohoku/pred_from_eta0` (coastal tsunami prediction from estimated source).
- Run `postprocess2.ipynb` to create the figures of the coastal tsunami prediction.
