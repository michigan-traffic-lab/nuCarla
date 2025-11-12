#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

wget -c https://huggingface.co/datasets/zhijieq/nuCarla/resolve/main/Town01.zip
wget -c https://huggingface.co/datasets/zhijieq/nuCarla/resolve/main/Town02.zip
wget -c https://huggingface.co/datasets/zhijieq/nuCarla/resolve/main/Town03.zip
wget -c https://huggingface.co/datasets/zhijieq/nuCarla/resolve/main/Town04.zip
wget -c https://huggingface.co/datasets/zhijieq/nuCarla/resolve/main/Town05.zip
wget -c https://huggingface.co/datasets/zhijieq/nuCarla/resolve/main/Town06.zip
wget -c https://huggingface.co/datasets/zhijieq/nuCarla/resolve/main/Town07.zip
wget -c https://huggingface.co/datasets/zhijieq/nuCarla/resolve/main/Town10.zip
wget -c https://huggingface.co/datasets/zhijieq/nuCarla/resolve/main/Mcity.zip
wget -c https://huggingface.co/datasets/zhijieq/nuCarla/resolve/main/Metadata.zip

echo "[INFO] All downloads completed successfully."