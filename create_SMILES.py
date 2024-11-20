from rdkit import Chem
import os


def create_smiles_in_dir(dir_path:str):
    
    for folder in os.listdir(dir_path):
        mol = Chem.MolFromMolFile(f"{dir_path}/{folder}/structure.mol")
        smiles = Chem.MolToSmiles(mol)
        with open(f"{dir_path}/{folder}/smiles.txt", "w") as file:
            file.write(smiles)



if __name__=="__main__":
    create_smiles_in_dir("./dataset")