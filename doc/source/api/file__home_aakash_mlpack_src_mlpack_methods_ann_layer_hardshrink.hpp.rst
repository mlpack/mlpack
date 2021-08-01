
.. _file__home_aakash_mlpack_src_mlpack_methods_ann_layer_hardshrink.hpp:

File hardshrink.hpp
===================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_ann_layer>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/hardshrink.hpp``)
--------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_hardshrink.hpp.rst



Detailed Description
--------------------

Lakshya Ojha
Same as soft thresholding, if its amplitude is smaller than a predefined threshold, it will be set to zero (kill), otherwise it will be kept unchanged. In order to promote sparsity and to improve the approximation, the hard thresholding method is used as an alternative.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``hardshrink_impl.hpp``

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


- :ref:`exhale_class_classmlpack_1_1ann_1_1HardShrink`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/ann/layer/hardshrink.hpp

