#!/bin/bash

echo Downloading and unzipping Youtube Mix dataset...
./google-drive-downloader.sh 1h0vaFsZDtJick-vrgBreRTq2p5DIHcAf youtube-mix.zip
unzip -o youtube-mix.zip

echo Downloading and unzipping pre-trained models...
./google-drive-downloader.sh 14H28L_vnZzTamuBsUUkkZ4gZDakxnkrA models-v1.zip
unzip -o models-v1.zip
