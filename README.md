#tsCluster

A Python framework for interactively examining time-series data collected in *runs* and *sessions* (example fMRI) from different classes/groups of objects (example tinnitus patients vs controls).

### Data-set
Data set is not included for confidentiality reasons. Import binaries.zip and data.zip from the DynamicConnectivity folder on the server.


## Instructions
We assume you have anaconda installed. There are two environment files.

  - ysjupyt.yml -- Has jupyter notebook service installed for interactive access
  - nojupyt.yml -- Has only the minimal amount of packages required to run on a cluster

### Interactive access

For interactive access create a conda environment using the `ysjupyt.yml` file. Then `cd` into the folder you want and run `jupyter notebook`. Open the `interactive.ipynb` notebook.

### Running on cluster

`scp` or use FileZilla to copy the `tsCluster` folder and `binaries` folder onto the scratch folder or home folder on the cluster. Import any `*.py` that you want to run and then submit the job using `qsub`.
