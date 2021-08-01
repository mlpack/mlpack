
.. _file__home_aakash_mlpack_src_mlpack_core_cereal_pointer_variant_wrapper.hpp:

File pointer_variant_wrapper.hpp
================================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_core_cereal>` (``/home/aakash/mlpack/src/mlpack/core/cereal``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/core/cereal/pointer_variant_wrapper.hpp``)
---------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_core_cereal_pointer_variant_wrapper.hpp.rst



Detailed Description
--------------------

Omar Shrit
Implementation of a boost::variant wrapper to enable the serialization of the pointers inside boost variant in cereal
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``boost/variant.hpp``

- ``boost/variant/static_visitor.hpp``

- ``boost/variant/variant_fwd.hpp``

- ``cereal/archives/json.hpp``

- ``cereal/archives/portable_binary.hpp``

- ``cereal/archives/xml.hpp``

- ``cereal/types/boost_variant.hpp``

- ``pointer_wrapper.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_cereal_pointer_wrapper.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_core_cereal_pointer_vector_variant_wrapper.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`




Namespaces
----------


- :ref:`namespace_cereal`


Classes
-------


- :ref:`exhale_struct_structcereal_1_1load__visitor`

- :ref:`exhale_struct_structcereal_1_1save__visitor`

- :ref:`exhale_class_classcereal_1_1PointerVariantWrapper`


Functions
---------


- :ref:`exhale_function_namespacecereal_1a45bbfc5cc5f47d0c1fcf9aaa1e613610`


Defines
-------


- :ref:`exhale_define_pointer__variant__wrapper_8hpp_1a8b900b2dd439187b5b190b71390c5731`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/core/cereal/pointer_variant_wrapper.hpp

