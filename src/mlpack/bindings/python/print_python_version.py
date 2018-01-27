#!/usr/bin/env python
#
# Print the generated installation directory that packages will be installed to,
# so that we can add it to the Python path from CMake.
#
# The argument to this should be the installation prefix.
import sys

print(sys.argv[1] + \
      '/lib/python' + \
      str(sys.version_info.major) + \
      '.' + \
      str(sys.version_info.minor) + \
      '/site-packages/')
