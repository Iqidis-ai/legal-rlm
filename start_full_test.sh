#!/bin/bash
cd "C:\Users\devan\OneDrive\Desktop\Projects\RLMs"
mkdir -p test_results
python run_full_test.py 2>&1 | tee test_results/full_output.log
