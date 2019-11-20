#!/usr/bin/env python3
import numpy as np
import sys

casename = sys.argv[1]

RMSEa = np.load(casename+"/RMSEa.npy")
CRa = np.load(casename+"/SPRDa.npy") / np.load(casename+"/RMSEa.npy")

print('    RMSEa = ', RMSEa)
print('      CRa = ', CRa)
