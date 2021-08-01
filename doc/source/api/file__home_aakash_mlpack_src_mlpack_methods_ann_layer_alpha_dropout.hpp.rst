
.. _file__home_aakash_mlpack_src_mlpack_methods_ann_layer_alpha_dropout.hpp:

File alpha_dropout.hpp
======================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_ann_layer>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/alpha_dropout.hpp``)
-----------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_alpha_dropout.hpp.rst



Detailed Description
--------------------

Dakshit Agrawal
Definition of the Alpha-Dropout class, which implements a regularizer that randomly sets units to alpha-dash to prevent them from co-adapting and makes an affine transformation so as to keep the mean and variance of outputs at their original values.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``alpha_dropout_impl.hpp``

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_layer_types.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_layer.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__ann`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1ann_1_1AlphaDropout`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/ann/layer/alpha_dropout.hpp

