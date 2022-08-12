#!/bin/bash

# https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
pip install gdown
gdown https://drive.google.com/uc?id=17d_z7OUmbywEu4VjRam1tnjQUtCFEl4A
unzip BatchComposites.zip
unlink BatchComposites.zip
python transform_data.py
rm -rf BatchComposites