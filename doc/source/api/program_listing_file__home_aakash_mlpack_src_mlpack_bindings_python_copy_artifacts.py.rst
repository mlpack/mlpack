
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_copy_artifacts.py:

Program Listing for File copy_artifacts.py
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_copy_artifacts.py>` (``/home/aakash/mlpack/src/mlpack/bindings/python/copy_artifacts.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

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
