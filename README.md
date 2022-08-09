### High level goals

This repository is dedicated specifically to the development of (Large) Language Models, and/or Language/Structure models in the bio-chem space. 



### Current plans
We'd like to start simple - developing some LLMs using SELFIES over (or with) SMILES, develop some baselines. Intended datasets are ZINC + PubChem.

Required Steps:

- [ ] Scripts for downloading ZINC & PubChem
- [ ] Scripts for converting ZINC/PubChem from SMILES to SELFIES, format TBD. (CSV of original columns (including SMILES) + selfies is probably fine for the moment)
- [ ] Decide on training harness for LLM, leaning towards something HF related for finetuning.
- [ ] Evaluation harness
- [ ] ... TBD.
