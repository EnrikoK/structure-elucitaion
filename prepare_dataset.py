import os
import re
import requests


SUBCULTURES_DATA_URL = "https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/"
DATA_DIR = "dataset"


def download_mol_file(path_param:str, dest_dir:str):
    response = requests.get(f"{SUBCULTURES_DATA_URL}{path_param}/{path_param}.mol", stream=True)
    with open(f"{dest_dir}/structure.mol", "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

def prepare_dataset(images_dir: str):

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    else:
        pass


    for image in os.listdir(images_dir):
        if image.startswith("(2D-HMBC)"):

            base_name = image.split(".")[0]

            os.mkdir(f"{DATA_DIR}/{base_name}")
            os.rename(f"{images_dir}/{image}",f"{DATA_DIR}/{base_name}/{image}")

            pattern = r"bmse\d+"
            matches = re.search(pattern, image)
            
            download_mol_file(matches.group(),f"{DATA_DIR}/{base_name}/")
            print(image)
        else:
            os.remove(f"{images_dir}/{image}")


if __name__=="__main__":
    prepare_dataset("./images")