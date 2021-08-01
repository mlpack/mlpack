
.. _file__home_aakash_mlpack_src_mlpack_methods_approx_kfn_drusilla_select.hpp:

File drusilla_select.hpp
========================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_approx_kfn>` (``/home/aakash/mlpack/src/mlpack/methods/approx_kfn``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/approx_kfn/drusilla_select.hpp``)
--------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_approx_kfn_drusilla_select.hpp.rst



Detailed Description
--------------------

Ryan Curtin
An implementation of the approximate furthest neighbor algorithm specified in the following paper:
@incollection{curtin2016fast,
title={Fastapproximatefurthestneighborswithdata-dependentcandidate
selection},
author={Curtin,R.R.,andGardner,A.B.},
booktitle={SimilaritySearchandApplications},
pages={221--235},
year={2016},
publisher={Springer}
}

This algorithm, called DrusillaSelect, constructs a candidate set of points to query to find an approximate furthest neighbor. The strange name is a result of the algorithm being named after a cat. The cat in question may be viewed at http://www.ratml.org/misc_img/drusilla_fence.png.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``drusilla_select_impl.hpp``

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)






Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__neighbor`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1neighbor_1_1DrusillaSelect`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/approx_kfn/drusilla_select.hpp

