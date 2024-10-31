
import os
import sys
from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt
import logging
import pdb

# Set up logging
PROCESS_NAME = "testNWB"
loglevel = 'INFO'
numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(format=f'[{PROCESS_NAME}] %(levelname)s: %(message)s',
                    level=numeric_level)

# Get NWB filepath from command line arguments
if len(sys.argv) < 2:
    logging.error(f"Insufficient number of arguments. Please provide the path to a NWB file generated from running testGraph")
    sys.exit(1)
filepath = sys.argv[1]

# Open NWB file
with NWBHDF5IO(filepath, "r") as io:
    read_nwbfile = io.read()
    
    # Optional: Python debugger to inspect the file
    #pdb.set_trace()
    
    # Extract function generator data, number of samples, and number of channels
    data = read_nwbfile.acquisition['func_generator'].data[:]
    n_samples = data.shape[0]
    n_channels = data.shape[1]

    logging.info(f"func_generator stream data has {n_samples} samples and {n_channels} channels")

FORMAT = 'png'
out_name = os.path.basename(filepath).split('.')[0] + '.' + FORMAT
out_dir = os.path.dirname(os.path.realpath(__file__))
out_path = os.path.join(out_dir,out_name)

nplots = 4
nsamps = 1000
logging.info(f"Plotting first {nsamps} samples for channels 0-{nplots}")

fig, axes = plt.subplots(ncols=1, nrows=nplots, figsize=(8, 12), sharey=True, sharex=True)
for i in range(0,nplots):
    axes[i].set_title(f"func_generator channel {i} data")
    axes[i].plot(data[:1000,i])
    axes[i].set_ylabel('Value (AU)')
axes[-1].set_xlabel('Sample #')    
plt.tight_layout()
plt.savefig(f'{out_path}')

logging.info(f"Saved plot to {out_path}")