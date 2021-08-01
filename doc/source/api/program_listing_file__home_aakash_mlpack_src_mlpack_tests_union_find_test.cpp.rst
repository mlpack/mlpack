
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_union_find_test.cpp:

Program Listing for File union_find_test.cpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_union_find_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/union_find_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/methods/emst/union_find.hpp>
   
   #include <mlpack/core.hpp>
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::emst;
   
   TEST_CASE("TestFind", "[UnionFindTest]")
   {
     static const size_t testSize = 10;
     UnionFind testUnionFind(testSize);
   
     for (size_t i = 0; i < testSize; ++i)
       REQUIRE(testUnionFind.Find(i) == i);
   
     testUnionFind.Union(0, 1);
     testUnionFind.Union(1, 2);
   
     REQUIRE(testUnionFind.Find(2) == testUnionFind.Find(0));
   }
   
   TEST_CASE("TestUnion", "[UnionFindTest]")
   {
     static const size_t testSize = 10;
     UnionFind testUnionFind(testSize);
   
     testUnionFind.Union(0, 1);
     testUnionFind.Union(2, 3);
     testUnionFind.Union(0, 2);
     testUnionFind.Union(5, 0);
     testUnionFind.Union(0, 6);
   
     REQUIRE(testUnionFind.Find(0) == testUnionFind.Find(1));
     REQUIRE(testUnionFind.Find(2) == testUnionFind.Find(3));
     REQUIRE(testUnionFind.Find(1) == testUnionFind.Find(5));
     REQUIRE(testUnionFind.Find(6) == testUnionFind.Find(3));
   }
