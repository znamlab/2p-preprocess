#!/usr/bin/env python3
import numpy as np
import sys

from suite2p import run_s2p, default_ops
from suite2p.extraction import masks

datapath = sys.argv[1]

# set your options for running
ops = default_ops()
ops['save_mat'] = True
db = []
db.append({'data_path': [datapath]})

opsEnd = run_s2p(ops=ops, db=db[0])
