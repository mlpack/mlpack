
.. _file__home_aakash_mlpack_src_mlpack_tests_tree_traits_test.cpp:

File tree_traits_test.cpp
=========================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_tests>` (``/home/aakash/mlpack/src/mlpack/tests``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/tests/tree_traits_test.cpp``)
--------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_tests_tree_traits_test.cpp.rst



Detailed Description
--------------------

Ryan Curtin
Tests for the TreeTraits class. These could all be known at compile-time, but realistically the function is to be sure that nobody changes tree traits without breaking something. Thus, people must be certain when they make a change like that (because they have to change the test too). That's the hope, at least...
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``catch.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_tests_catch.hpp`)

- ``mlpack/core.hpp``

- ``mlpack/core/tree/binary_space_tree.hpp``

- ``mlpack/core/tree/cover_tree.hpp``

- ``mlpack/core/tree/rectangle_tree.hpp``

- ``mlpack/core/tree/tree_traits.hpp``

- ``test_catch_tools.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_tests_test_catch_tools.hpp`)






Functions
---------


- :ref:`exhale_function_tree__traits__test_8cpp_1ada5691aad63be496f4f4a69d9a83c5fe`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/tests/tree_traits_test.cpp

