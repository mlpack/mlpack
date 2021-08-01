
.. _file__home_aakash_mlpack_src_mlpack_bindings_python_get_arma_type.hpp:

File get_arma_type.hpp
======================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_bindings_python>` (``/home/aakash/mlpack/src/mlpack/bindings/python``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/bindings/python/get_arma_type.hpp``)
---------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_get_arma_type.hpp.rst



Detailed Description
--------------------

Ryan Curtin
Return "mat", "col", or "row" depending on the type of the given Armadillo object. This is so that the correct overload of arma_numpy.numpy_to_<type>() can be called.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_bindings_python_print_input_processing.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_bindings_python_print_output_processing.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__bindings`

- :ref:`namespace_mlpack__bindings__python`


Functions
---------


- :ref:`exhale_function_namespacemlpack_1_1bindings_1_1python_1a6e0e0614e11b883601227d5bf884fce6`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/bindings/python/get_arma_type.hpp

