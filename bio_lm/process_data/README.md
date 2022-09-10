# Data Processing Scripts

This hosts the scripts used to process the data for the language model.
Currently, it uses DeepChem to get the ZINC dataset and convert SMILES to SELFIES.

On the 200K samples, it takes ~1min to process. 

To process the data run

```bash
python process_data.py --size 250k --dir data
```
