#!/bin/bash

# Parse arguments: --config (or -c) --dataset (or -d)
while [[ $# -gt 0 ]];
do
    case $1 in
        -c|--config)
            CONFIG_FILE=$2
            shift
            shift
            ;;
        -d|--dataset)
            DATASET_FILE=$2
            shift
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done
# Check if required arguments are provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Both --config and --dataset arguments are required."
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE does not exist."
    exit 1
fi

# Check if dataset file exists if provided
if [ -n "$DATASET_FILE" ] && [ ! -f "$DATASET_FILE" ]; then
    echo "Error: Dataset file $DATASET_FILE does not exist."
    exit 1
fi
