
.. _file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_oivs_init.hpp:

File oivs_init.hpp
==================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_ann_init_rules>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/oivs_init.hpp``)
------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_oivs_init.hpp.rst



Detailed Description
--------------------

Marcus Edel
Definition and implementation of the Optimal Initial Value Setting method (OIVS). This initialization rule is based on geometrical considerations as described by H. Shimodaira.
For more information, see the following paper.
@inproceedings{ShimodairaICTAI1994,
title={Aweightvalueinitializationmethodforimprovinglearning
performanceofthebackpropagationalgorithminneuralnetworks},
author={Shimodaira,H.},
booktitle={ToolswithArtificialIntelligence,1994.Proceedings.,
SixthInternationalConferenceon},
year={1994}
}

mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``mlpack/methods/ann/activation_functions/logistic_function.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_logistic_function.hpp`)

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)

- ``random_init.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_amf_init_rules_random_init.hpp`)






Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__ann`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1ann_1_1OivsInitialization`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/ann/init_rules/oivs_init.hpp

