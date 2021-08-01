
.. _file__home_aakash_mlpack_src_mlpack_core_dists_gamma_distribution.hpp:

File gamma_distribution.hpp
===========================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_core_dists>` (``/home/aakash/mlpack/src/mlpack/core/dists``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/core/dists/gamma_distribution.hpp``)
---------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_core_dists_gamma_distribution.hpp.rst



Detailed Description
--------------------

Yannis Mentekidis 
Rohan Raj
Implementation of a Gamma distribution of multidimensional data that fits gamma parameters (alpha, beta) to data. The fitting is done independently for each dataset dimension (row), based on the assumption each dimension is fully independent.
Based on "Estimating a Gamma Distribution" by Thomas P. Minka: research.microsoft.com/~minka/papers/minka-gamma.pdf
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``mlpack/core/math/random.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_math_random.hpp`)

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_core.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__distribution`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1distribution_1_1GammaDistribution`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/core/dists/gamma_distribution.hpp

