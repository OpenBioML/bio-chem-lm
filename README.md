# High level goals

This repository is dedicated specifically to the development of (Large) Language Models, and/or Language/Structure models in the bio-chem space. 

Further details can be found [here](https://docs.google.com/document/d/1HFfZY3RjbRN9SfHJIcxvsA8oTk_XFY7dXmiICnbcUVU/edit)

## Bio-LM PubChem Selfies

We are training an Electra-style model on the PubChem dataset with SELFIES representations. The SELFIES  is a chemical language that is based on the SMILES language, but is more robust. More info about SELFIES can be found [here](https://github.com/aspuru-guzik-group/selfies).

We have released the dataset to [HuggingFace Datasets](https://huggingface.co/datasets/zpn/pubchem_selfies), which contains ~110M compounds in total.

We will perform a hyperparameter search using [Maximal Update Parameterization](https://github.com/microsoft/mup) to find a good set of hyperparameters to transfer to a larger model. To launch a sweep on the cluster, run

```bash
sbatch --array=1-N mup_train.sh
```