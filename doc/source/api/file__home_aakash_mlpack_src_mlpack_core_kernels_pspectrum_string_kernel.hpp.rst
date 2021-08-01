
.. _file__home_aakash_mlpack_src_mlpack_core_kernels_pspectrum_string_kernel.hpp:

File pspectrum_string_kernel.hpp
================================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_core_kernels>` (``/home/aakash/mlpack/src/mlpack/core/kernels``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/core/kernels/pspectrum_string_kernel.hpp``)
----------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_pspectrum_string_kernel.hpp.rst



Detailed Description
--------------------

Ryan Curtin
Implementation of the p-spectrum string kernel, created for use with FastMKS. Instead of passing a data matrix to FastMKS which stores the kernels, pass a one-dimensional data matrix (data vector) to FastMKS which stores indices of strings; then, the actual strings are given to the PSpectrumStringKernel at construction time, and the kernel knows to map the indices to actual strings.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``map`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_data_dataset_mapper.hpp`)

- ``mlpack/core/util/log.hpp``

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)

- ``pspectrum_string_kernel_impl.hpp``

- ``string`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_policies_bag_of_words_encoding_policy.hpp`)

- ``vector`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_util_is_std_vector.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_core.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_fastmks_fastmks_model.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__kernel`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1kernel_1_1PSpectrumStringKernel`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/core/kernels/pspectrum_string_kernel.hpp

