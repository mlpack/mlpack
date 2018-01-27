#!/usr/bin/env python
#
# Copy all built artifacts from build/lib.*/* to mlpack/.
import sys
import sysconfig
import shutil
import os

directory = 'build/lib.' + \
            sysconfig.get_platform() + \
            '-' + \
            str(sys.version_info[0]) + \
            '.' + \
            str(sys.version_info[1]) + \
            '/mlpack/'

# Now copy all the files from the directory to the desired location.
for f in os.listdir(directory):
  shutil.copy(directory + f, 'mlpack/' + f)
