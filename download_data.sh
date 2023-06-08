#!/bin/bash

echo Downloading and unzipping the Youtube Mix models...
wget https://huggingface.co/necrashter/SaShiMi-796/resolve/main/youtube-mix-models.zip -q --show-progress
unzip -o youtube-mix-models.zip

echo Downloading and unzipping the MNIST models...
wget https://huggingface.co/necrashter/SaShiMi-796/resolve/main/mnist-models.zip -q --show-progress
unzip -o mnist-models.zip

echo
echo Downloading and unzipping the Youtube Mix dataset...
wget https://huggingface.co/necrashter/SaShiMi-796/resolve/main/youtube-mix.zip -q --show-progress
unzip -o youtube-mix.zip

echo Done!
echo You can now safely delete the downloaded zip files to save storage space.
