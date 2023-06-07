#!/bin/bash

echo Downloading and unzipping the models...
wget https://huggingface.co/necrashter/SaShiMi-796/resolve/main/youtube-mix-models.zip -q --show-progress
unzip -o youtube-mix-models.zip

echo
echo Downloading and unzipping the Youtube Mix dataset...
wget https://huggingface.co/necrashter/SaShiMi-796/resolve/main/youtube-mix.zip -q --show-progress
unzip -o youtube-mix.zip

echo Done!
