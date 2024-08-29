A Python package for computing opacity cross-sections, using line lists from VALD, ExoMol or HITRAN/HITEMP. 

# Usage
Give the input python-file as the first positional argument when calling `main.py` (see examples of the input files in `input/`).

## Computing cross-sections
```
python main.py input/XXX -cs
```
or to run for transition-files between `i_min=0` and `i_max=10` (for large line lists):
```
python main.py input/XXX -cs -i_min 0 -i_max 10
```
Each call with `-cs` will create temporary `.hdf5` files of the considered transitions. That way, separate chunks of transition-files can be run in parallel. The temporary files need to be combined to the final output using:
```
python main.py input/XXX -s
```

## Appending to existing save
```
python main.py input/XXX -s --append_to_existing
```
will add new pressure or temperature points to an existing `.hdf5` save (into a new `*_new.hdf5` to avoid overwriting). (**can only handle expansion of pressure- or temperature-axis at a time**)

## Converting to pRT2-format
```
python main.py input/XXX --convert_to_pRT2
```
will create the `short_stream/` opacity-directory for petitRADTRANS 2. 

## Plotting
```
python main.py input/XXX --plot
```
or (to plot the pRT-converted opacities):
```
python figures.py
```

## Downloading data (ExoMol+HITRAN/HITEMP)
```
python main.py input/XXX --download
```
for ExoMol, will download the `*.json` definition file, all transitions (`*.trans.bz2`), states (`*.states.bz2`) and partition function (`*.pf`) (see `inputs/example_ExoMol_NaH.py`). For HITRAN/HITEMP, will download any files in the `urls` list given in the input (see `inputs/example_HITEMP.py`). 