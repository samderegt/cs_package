A Python package for computing opacity cross-sections, using line lists from VALD, ExoMol or HITRAN/HITEMP. 

## Usage
In `main.py`, modify `import input.XXX as conf` to read in the correct inputs (see examples in `input/`).

#### Computing cross-sections
```
python main.py -cs
```
or to run for transition-files between `i_min=0` and `i_max=10` (for large line lists):
```
python main.py -cs -i_min 0 -i_max 10
```
Each call with `-cs` will create temporary `.hdf5` files of the considered transitions. That way, separate chunks of transition-files can be run in parallel. The temporary files need to be combined to the final output using:
```
python main.py -s
```

#### Appending to existing save
```
python main.py -s --append_to_existing
```
will add new pressure or temperature points to an existing `.hdf5` save (into a new `*_new.hdf5` to avoid overwriting). (**can only handle expansion of pressure- or temperature-axis at a time**)


#### Converting to pRT2-format
```
python main.py --convert_to_pRT2
```
will create the `short_stream/` opacity-directory for petitRADTRANS 2. 

#### Plotting
```
python main.py --plot
```
or (to plot the pRT-converted opacities):
```
python figures.py
```

#### Downloading data (ExoMol only)
```
python main.py --download
```
will download the `*.json` definition file, all transitions (`*.trans.bz2`), states (`*.states.bz2`) and partition function (`*.pf`) from the ExoMol database (see `inputs/example_ExoMol.py`). 