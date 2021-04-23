#!/usr/bin/env python3
import numpy as np
import sys

from suite2p import run_s2p, default_ops

# set your options for running
ops = default_ops()

db = []
db.append({'data_path': [sys.argv[1]]})

opsEnd = run_s2p(ops=ops, db=db)
