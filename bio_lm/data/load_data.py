import deepchem as dc
from deepchem.molnet import load_zinc15
import selfies as sf


def load_dataset():
    featurizer = dc.feat.RawFeaturizer(smiles=True)
    tasks, datasets, transformer = load_zinc15(featurizer=featurizer)
    train, valid, test = datasets

    return tasks, (train, valid, test), transformer

    
def convert_to_selfies(smile):
    return sf.encoder(smile) 

    
if __name__ == '__main__':
    tasts, (train, valid, test), transformer = load_dataset()

    for smile in train.X:
        print(smile)
        print(convert_to_selfies(smile))
        break

