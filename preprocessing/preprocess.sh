#!/bin/bash

python3 preprocessing.py
python3 to_phoneme.py
python3 token_alignment.py