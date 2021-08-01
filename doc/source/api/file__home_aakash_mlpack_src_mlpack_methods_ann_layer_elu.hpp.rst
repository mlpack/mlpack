
.. _file__home_aakash_mlpack_src_mlpack_methods_ann_layer_elu.hpp:

File elu.hpp
============

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_ann_layer>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/elu.hpp``)
-------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_elu.hpp.rst



Detailed Description
--------------------

Vivek Pal 
Dakshit Agrawal
Definition of the ELU activation function as described by Djork-Arne Clevert, Thomas Unterthiner and Sepp Hochreiter.
Definition of the SELU function as introduced by Klambauer et. al. in Self Neural Networks. The SELU activation function keeps the mean and variance of the input invariant.
In short, SELU = lambda * ELU, with 'alpha' and 'lambda' fixed for normalized inputs.
Hence both ELU and SELU are implemented in the same file, with lambda = 1 for ELU function.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``elu_impl.hpp``

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


- :ref:`exhale_class_classmlpack_1_1ann_1_1ELU`


Typedefs
--------


- :ref:`exhale_typedef_namespacemlpack_1_1ann_1ac08f9682be904369ec09e68b43b09fad`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/ann/layer/elu.hpp

