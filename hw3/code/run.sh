#!/bin/bash
python 4gramsLM.py --type Linear  --hidden 128
python 4gramsLM.py --type Linear  --hidden 256
python 4gramsLM.py --type Linear  --hidden 512
python 4gramsLM.py --type tanh  --hidden 128