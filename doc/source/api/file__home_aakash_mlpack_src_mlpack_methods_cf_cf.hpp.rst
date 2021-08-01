
.. _file__home_aakash_mlpack_src_mlpack_methods_cf_cf.hpp:

File cf.hpp
===========

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_cf>` (``/home/aakash/mlpack/src/mlpack/methods/cf``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/cf/cf.hpp``)
-----------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_cf.hpp.rst



Detailed Description
--------------------

Mudit Raj Gupta 
Sumedh Ghaisas
Collaborative filtering.
Defines the CFType class to perform collaborative filtering on the specified data set using alternating least squares (ALS).
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``cf_impl.hpp``

- ``iostream``

- ``map`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_data_dataset_mapper.hpp`)

- ``mlpack/methods/amf/amf.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_amf_amf.hpp`)

- ``mlpack/methods/amf/termination_policies/simple_residue_termination.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_simple_residue_termination.hpp`)

- ``mlpack/methods/amf/update_rules/nmf_als.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_nmf_als.hpp`)

- ``mlpack/methods/cf/decomposition_policies/nmf_method.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_cf_decomposition_policies_nmf_method.hpp`)

- ``mlpack/methods/cf/interpolation_policies/average_interpolation.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_cf_interpolation_policies_average_interpolation.hpp`)

- ``mlpack/methods/cf/neighbor_search_policies/lmetric_search.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_cf_neighbor_search_policies_lmetric_search.hpp`)

- ``mlpack/methods/cf/normalization/no_normalization.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_no_normalization.hpp`)

- ``mlpack/methods/neighbor_search/neighbor_search.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_neighbor_search.hpp`)

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)

- ``set`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_bias_set_visitor.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_methods_bias_svd_bias_svd.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_cf_cf_model.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_regularized_svd_regularized_svd.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_svdplusplus_svdplusplus.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__cf`


Classes
-------


- :ref:`exhale_struct_structmlpack_1_1cf_1_1CFType_1_1CandidateCmp`

- :ref:`exhale_class_classmlpack_1_1cf_1_1CFType`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/cf/cf.hpp

