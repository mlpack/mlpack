
.. _file__home_aakash_mlpack_src_mlpack_core_data_normalize_labels.hpp:

File normalize_labels.hpp
=========================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_core_data>` (``/home/aakash/mlpack/src/mlpack/core/data``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/core/data/normalize_labels.hpp``)
------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_core_data_normalize_labels.hpp.rst



Detailed Description
--------------------

Ryan Curtin
Often labels are not given as {0, 1, 2, ...} but instead {1, 2, ...} or even {-1, 1} or otherwise. The purpose of this function is to normalize labels to {0, 1, 2, ...} and provide a mapping back to those labels.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)

- ``normalize_labels_impl.hpp``



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_core.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__data`


Functions
---------


- :ref:`exhale_function_namespacemlpack_1_1data_1a664b3fa5243889e2aed47ee750f840ed`

- :ref:`exhale_function_namespacemlpack_1_1data_1a901fe08dcdc58734f64a864dbdef0a28`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/core/data/normalize_labels.hpp

