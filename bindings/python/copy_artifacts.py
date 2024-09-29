#!/usr/bin/env python
#
# Copy all built artifacts from build/lib.*/* to mlpack/.
#
# mlpack is free software; you may redistribute it and/or modify it under the
# terms of the 3-clause BSD license.  You should have received a copy of the
# 3-clause BSD license along with mlpack.  If not, see
# http://www.opensource.org/licenses/BSD-3-Clause for more information.
import sysconfig
import shutil
import os
import glob

# Match any lib.$platform*/mlpack/ directory.
directory = glob.glob('build/lib.' + sysconfig.get_platform() + '*/mlpack/')[0]
directory = directory.replace('\\', '/')

# Now copy all the files from the directory to the desired location.
for f in os.listdir(directory):
  shutil.copy(directory + f, 'mlpack/' + f)
