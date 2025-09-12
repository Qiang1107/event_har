#!/bin/bash

# Array of time window parameters
time_windows=(0.333 0.5 0.8 1.0 1.5 2.0)

# Loop through each time window parameter
for window in "${time_windows[@]}"; do
    echo "Running prediction_voting_withoutmix.py with time window: ${window} seconds"
    python prediction_voting_withoutmix.py --time_window ${window}
    echo "Completed time window: ${window} seconds"
    echo "----------------------------------------"
done

echo "All temporal voting tests completed."