
.. _file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_delete_visitor.hpp:

File delete_visitor.hpp
=======================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_ann_visitor>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/delete_visitor.hpp``)
--------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_delete_visitor.hpp.rst



Detailed Description
--------------------

Marcus Edel
This file provides an abstraction for the Delete() function for different layers and automatically directs any parameter to the right layer type.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``boost/variant.hpp``

- ``delete_visitor_impl.hpp``

- ``mlpack/methods/ann/layer/layer_traits.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_layer_traits.hpp`)

- ``mlpack/methods/ann/layer/layer_types.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_layer_types.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_brnn.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_ffn.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_add_merge.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concat.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_sequential.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_highway.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_merge.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_recurrent.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_weight_norm.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_rnn.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__ann`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1ann_1_1DeleteVisitor`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/ann/visitor/delete_visitor.hpp

