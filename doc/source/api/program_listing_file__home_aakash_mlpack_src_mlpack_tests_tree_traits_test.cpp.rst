
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_tree_traits_test.cpp:

Program Listing for File tree_traits_test.cpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_tree_traits_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/tree_traits_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/core/tree/tree_traits.hpp>
   #include <mlpack/core/tree/binary_space_tree.hpp>
   #include <mlpack/core/tree/cover_tree.hpp>
   #include <mlpack/core/tree/rectangle_tree.hpp>
   
   #include "catch.hpp"
   #include "test_catch_tools.hpp"
   
   using namespace mlpack;
   using namespace mlpack::tree;
   using namespace mlpack::metric;
   
   // Be careful!  When writing new tests, always get the boolean value of each
   // trait and store it in a temporary, because the Boost unit test macros do
   // weird things and will cause bizarre problems.
   
   // Test the defaults.
   TEST_CASE("DefaultsTraitsTest", "[TreeTraitsTest]")
   {
     // An irrelevant non-tree type class is used here so that the default
     // implementation of TreeTraits is chosen.
     bool b = TreeTraits<int>::HasOverlappingChildren;
     REQUIRE(b == true);
     b = TreeTraits<int>::HasSelfChildren;
     REQUIRE(b == false);
     b = TreeTraits<int>::FirstPointIsCentroid;
     REQUIRE(b == false);
     b = TreeTraits<int>::RearrangesDataset;
     REQUIRE(b == false);
     b = TreeTraits<int>::BinaryTree;
     REQUIRE(b == false);
   }
   
   // Test the binary space tree traits.
   TEST_CASE("BinarySpaceTreeTraitsTest", "[TreeTraitsTest]")
   {
     typedef BinarySpaceTree<LMetric<2, false>> TreeType;
   
     // Children are non-overlapping.
     bool b = TreeTraits<TreeType>::HasOverlappingChildren;
     REQUIRE(b == false);
   
     // Points are not contained at multiple levels.
     b = TreeTraits<TreeType>::HasSelfChildren;
     REQUIRE(b == false);
   
     // The first point is not the centroid.
     b = TreeTraits<TreeType>::FirstPointIsCentroid;
     REQUIRE(b == false);
   
     // The dataset gets rearranged at build time.
     b = TreeTraits<TreeType>::RearrangesDataset;
     REQUIRE(b == true);
   
     // It is a binary tree.
     b = TreeTraits<TreeType>::BinaryTree;
     REQUIRE(b == true);
   }
   
   // Test the cover tree traits.
   TEST_CASE("CoverTreeTraitsTest", "[TreeTraitsTest]")
   {
     // Children may be overlapping.
     bool b = TreeTraits<CoverTree<>>::HasOverlappingChildren;
     REQUIRE(b == true);
   
     // The cover tree has self-children.
     b = TreeTraits<CoverTree<>>::HasSelfChildren;
     REQUIRE(b == true);
   
     // The first point is the center of the node.
     b = TreeTraits<CoverTree<>>::FirstPointIsCentroid;
     REQUIRE(b == true);
   
     b = TreeTraits<CoverTree<>>::RearrangesDataset;
     REQUIRE(b == false);
   
     b = TreeTraits<CoverTree<>>::BinaryTree;
     REQUIRE(b == false); // Not necessarily binary.
   }
