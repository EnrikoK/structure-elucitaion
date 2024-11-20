# SMILES structure elucitation from 2D NMR images

### Data

The 2D NMR images were taken from: https://github.com/stefhk3/nmrchemclassify/tree/main/images

Based on the individual images, the .mol file was queried from https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/ and converted to a SMILES string that was saved as smiles.txt

folder structure example

```
images/
  (2D-HMBC)_structure_name/
    (2D-HMBC)_structure_name.png
    structure.mol
    smiles.txt
  ...
```
