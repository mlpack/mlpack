
.. _file__home_aakash_mlpack_src_mlpack_core_util_arma_config_check.hpp:

File arma_config_check.hpp
==========================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_core_util>` (``/home/aakash/mlpack/src/mlpack/core/util``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/core/util/arma_config_check.hpp``)
-------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_core_util_arma_config_check.hpp.rst



Detailed Description
--------------------

Ryan Curtin
Using the contents of arma_config.hpp, try to catch the condition where the user has included mlpack with ARMA_64BIT_WORD enabled but mlpack was compiled without ARMA_64BIT_WORD enabled. This should help prevent a long, drawn-out debugging process where nobody can figure out why the stack is getting mangled.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``arma_config.hpp``



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`




Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/core/util/arma_config_check.hpp

