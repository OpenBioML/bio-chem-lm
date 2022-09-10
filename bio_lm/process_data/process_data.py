import deepchem as dc
from deepchem.molnet import load_zinc15
import selfies as sf
import argparse
from tqdm import tqdm
from enum import Enum
from pathlib import Path


tqdm.pandas()

# this could prob be done better
class DatasetName(Enum):
    train = 0 
    valid = 1
    test = 2


def load_dataset(size="250K"):
    # just load raw data with smiles
    featurizer = dc.feat.RawFeaturizer(smiles=True)
    # datasets is a tuple of train, valid, test
    tasks, datasets, transformer = load_zinc15(featurizer=featurizer)
    train, valid, test = datasets

    return tasks, (train, valid, test), transformer

    
def smiles_to_selfies(smile):
    return sf.encoder(smile) 

def create_dict(smiles):
    selfies = smiles_to_selfies(smiles)
    d = {"smiles": smiles, "selfies": selfies}
    return d


def process_data(dataset, path):
    dataset = dataset.to_dataframe()
    dataset = dataset.rename(columns={"X": "smile"})

    dataset["selfie"] = dataset["smile"].progress_apply(smiles_to_selfies)

    # TODO: do we want to store everything or just selfie, smiles, id?
    dataset.to_json(path, orient="records", lines=True)

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="250K")
    parser.add_argument("--dir", type=str, default="data/", help="directory to save data")
    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    args = parse_args()
    tasks, datasets, transformer = load_dataset(size=args.size)

    folder = Path(args.dir)
    if not folder.exists():
        folder.mkdir(parents=True)

    for i, d in enumerate(datasets):
        name = DatasetName(i).name
        path = f"{args.dir}{name}.jsonl"
        process_data(d, path)