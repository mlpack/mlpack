
.. _file__home_aakash_mlpack_src_mlpack_methods_lsh_lsh_search.hpp:

File lsh_search.hpp
===================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_lsh>` (``/home/aakash/mlpack/src/mlpack/methods/lsh``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/lsh/lsh_search.hpp``)
--------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_lsh_lsh_search.hpp.rst



Detailed Description
--------------------

Parikshit Ram
Defines the LSHSearch class, which performs an approximate nearest neighbor search for a queries in a query set over a given dataset using Locality-sensitive hashing with 2-stable distributions.
The details of this method can be found in the following paper:
@inproceedings{datar2004locality,
title={Locality-sensitivehashingschemebasedonp-stabledistributions},
author={Datar,M.andImmorlica,N.andIndyk,P.andMirrokni,V.S.},
booktitle=
{Proceedingsofthe12thAnnualSymposiumonComputationalGeometry},
pages={253--262},
year={2004},
organization={ACM}
}

Additionally, the class implements Multiprobe LSH, which improves approximation results during the search for approximate nearest neighbors. The Multiprobe LSH algorithm was presented in the paper:
@inproceedings{Lv2007multiprobe,
tile={Multi-probeLSH:efficientindexingforhigh-dimensionalsimilarity
search},
author={Lv,QinandJosephson,WilliamandWang,ZheandCharikar,Mosesand
Li,Kai},
booktitle={Proceedingsofthe33rdinternationalconferenceonVerylarge
databases},
year={2007},
pages={950--961}
}

mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``lsh_search_impl.hpp``

- ``mlpack/core/metrics/lmetric.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_metrics_lmetric.hpp`)

- ``mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_sort_policies_nearest_neighbor_sort.hpp`)

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)

- ``queue``






Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__neighbor`


Classes
-------


- :ref:`exhale_struct_structmlpack_1_1neighbor_1_1LSHSearch_1_1CandidateCmp`

- :ref:`exhale_class_classmlpack_1_1neighbor_1_1LSHSearch`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/lsh/lsh_search.hpp

