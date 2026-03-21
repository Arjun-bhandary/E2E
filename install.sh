#!/bin/bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install spconv-cu120==2.3.6

pip install -r requirements.txt
