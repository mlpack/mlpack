
.. _file__home_aakash_mlpack_src_mlpack_core_tree_cellbound.hpp:

File cellbound.hpp
==================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_core_tree>` (``/home/aakash/mlpack/src/mlpack/core/tree``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/core/tree/cellbound.hpp``)
-----------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_cellbound.hpp.rst



Detailed Description
--------------------

Mikhail Lozhnikov
Definition of the CellBound class. The class describes a bound that consists of a number of hyperrectangles. These hyperrectangles do not overlap each other. The bound is limited by an outer hyperrectangle and two addresses, the lower address and the high address. Thus, the bound contains all points included between the lower and the high addresses.
The notion of addresses is described in the following paper. @inproceedings{bayer1997,
author={Bayer,Rudolf},
title={TheUniversalB-TreeforMultidimensionalIndexing:General
Concepts},
booktitle={ProceedingsoftheInternationalConferenceonWorldwide
ComputingandItsApplications},
series={WWCA'97},
year={1997},
isbn={3-540-63343-X},
pages={198--209},
numpages={12},
publisher={Springer-Verlag},
address={London,UK,UK},
}

mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``address.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_address.hpp`)

- ``bound_traits.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_bound_traits.hpp`)

- ``cellbound_impl.hpp``

- ``mlpack/core/math/range.hpp``

- ``mlpack/core/metrics/lmetric.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_metrics_lmetric.hpp`)

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_core_tree_bounds.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__bound`


Classes
-------


- :ref:`exhale_struct_structmlpack_1_1bound_1_1BoundTraits_3_01CellBound_3_01MetricType_00_01ElemType_01_4_01_4`

- :ref:`exhale_class_classmlpack_1_1bound_1_1CellBound`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/core/tree/cellbound.hpp

