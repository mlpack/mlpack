
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_print_python_version.py:

Program Listing for File print_python_version.py
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_print_python_version.py>` (``/home/aakash/mlpack/src/mlpack/bindings/python/print_python_version.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

   #!/usr/bin/env python
   #
   # Print the generated installation directory that packages will be installed to,
   # so that we can add it to the Python path from CMake.
   #
   # The argument to this should be the installation prefix.
   #
   # mlpack is free software; you may redistribute it and/or modify it under the
   # terms of the 3-clause BSD license.  You should have received a copy of the
   # 3-clause BSD license along with mlpack.  If not, see
   # http://www.opensource.org/licenses/BSD-3-Clause for more information.
   import sys
   
   print(sys.argv[1] + \
         '/lib/python' + \
         str(sys.version_info.major) + \
         '.' + \
         str(sys.version_info.minor) + \
         '/site-packages/')
