
.. _file__home_aakash_mlpack_src_mlpack_methods_lars_lars.hpp:

File lars.hpp
=============

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_lars>` (``/home/aakash/mlpack/src/mlpack/methods/lars``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/lars/lars.hpp``)
---------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_lars_lars.hpp.rst



Detailed Description
--------------------

Nishant Mehta (niche)
Definition of the LARS class, which performs Least Angle Regression and the LASSO.
Only minor modifications of LARS are necessary to handle the constrained version of the problem:
:math:`[ \min_{\beta} 0.5 || X \beta - y ||_2^2 + 0.5 \lambda_2 || \beta ||_2^2 \` subject to :math:` ||\beta||_1 <= \tau `
Although this option currently is not implemented, it will be implemented very soon.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``lars_impl.hpp``

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_methods_local_coordinate_coding_lcc.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_sparse_coding.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__regression`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1regression_1_1LARS`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/lars/lars.hpp

