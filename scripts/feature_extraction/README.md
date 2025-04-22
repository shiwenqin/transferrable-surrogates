# Extracting the ZCP and GRAF features

This script extracts the zero-cost proxies and GRAF features from the saved individuals from the evolutionary runs. There is one file for each run and seed named `data_<dataset>_seed=<seed>.pkl`. The file should contain the whole log for the given dataset - a dictionary with key `rewards` that contains a list of individuals `ind` that have the architecture at index `ind[0][0]` and the accuracy at `ind[1]`. 

The script expects to be in the main einsearch directory, with einsearch and its dependencies installed. The datasets should be `../data`, the pickle files with the lists of individuals should be in the `evolution_data` directory in the same place as the script. The script will produce files `<dataset>_seed=<seed>.csv` with the extracted GRAF and ZCPs. Run the script with an einsearch config for the same dataset from which the features should be extracted (the script only uses the dataset name and its metadata for its run, the rest of the settings is irrelevant).

