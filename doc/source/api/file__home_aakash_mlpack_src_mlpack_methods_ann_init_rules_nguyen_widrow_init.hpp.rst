
.. _file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_nguyen_widrow_init.hpp:

File nguyen_widrow_init.hpp
===========================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_ann_init_rules>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp``)
---------------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_nguyen_widrow_init.hpp.rst



Detailed Description
--------------------

Marcus Edel
Definition and implementation of the Nguyen-Widrow method. This initialization rule initialize the weights so that the active regions of the neurons are approximately evenly distributed over the input space.
For more information, see the following paper.
@inproceedings{NguyenIJCNN1990,
title={Improvingthelearningspeedof2-layerneuralnetworksbychoosing
initialvaluesoftheadaptiveweights},
booktitle={NeuralNetworks,1990.,1990IJCNNInternationalJoint
Conferenceon},
year={1990}
}

mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``init_rules_traits.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_init_rules_traits.hpp`)

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)

- ``random_init.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_amf_init_rules_random_init.hpp`)






Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__ann`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1ann_1_1InitTraits_3_01NguyenWidrowInitialization_01_4`

- :ref:`exhale_class_classmlpack_1_1ann_1_1NguyenWidrowInitialization`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp

