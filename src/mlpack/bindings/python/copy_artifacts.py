#!/usr/bin/env python
#
# Copy all built artifacts from build/lib.*/* to mlpack/.
#
# mlpack is free software; you may redistribute it and/or modify it under the
# terms of the 3-clause BSD license.  You should have received a copy of the
# 3-clause BSD license along with mlpack.  If not, see
# http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
