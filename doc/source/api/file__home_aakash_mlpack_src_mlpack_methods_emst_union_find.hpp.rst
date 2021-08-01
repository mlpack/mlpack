
.. _file__home_aakash_mlpack_src_mlpack_methods_emst_union_find.hpp:

File union_find.hpp
===================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_emst>` (``/home/aakash/mlpack/src/mlpack/methods/emst``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/emst/union_find.hpp``)
---------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_emst_union_find.hpp.rst



Detailed Description
--------------------

Bill March (march@gatech.edu)
Implements a union-find data structure. This structure tracks the components of a graph. Each point in the graph is initially in its own component. Calling unionfind.Union(x, y) unites the components indexed by x and y. unionfind.Find(x) returns the index of the component containing point x.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_methods_dbscan_dbscan.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_emst_edge_pair.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__emst`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1emst_1_1UnionFind`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/emst/union_find.hpp

