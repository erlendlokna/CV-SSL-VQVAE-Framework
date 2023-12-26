import os
import requests
import tarfile
from pathlib import Path
from utils import get_root_dir


def download_ucr_datasets(url='https://figshare.com/ndownloader/files/37909926', chunk_size=128, zip_fname='UCR_archive.zip'):
    #dirname = str(get_root_dir().joinpath("data"))
    dirname = './data'
    if os.path.isdir(os.path.join(dirname, 'UCRArchive_2018')):
        return None

    if not os.path.isdir(dirname) or not os.path.isfile(os.path.join(dirname, zip_fname)):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # download
        r = requests.get(url, stream=True)
        print('downloading the UCR archive datasets...\n')
        fname = os.path.join(dirname, zip_fname)
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

        # unzip
        with tarfile.open(fname) as ff:
            ff.extractall(path=dirname)
    elif os.path.isfile(str(get_root_dir().joinpath("datasets", zip_fname))):
        # unzip
        fname = os.path.join(dirname, zip_fname)
        with tarfile.open(fname) as ff:
            ff.extractall(path=dirname)
    else:
        pass

    os.remove('./data/UCR_archive.zip')

if __name__ == "__main__":
    download_ucr_datasets()