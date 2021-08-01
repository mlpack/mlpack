
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_tree_traits.hpp:

Program Listing for File tree_traits.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_tree_traits.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/tree_traits.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_TREE_TRAITS_HPP
   #define MLPACK_CORE_TREE_TREE_TRAITS_HPP
   
   namespace mlpack {
   namespace tree {
   
   template<typename TreeType>
   class TreeTraits
   {
    public:
     static const bool HasOverlappingChildren = true;
   
     static const bool HasDuplicatedPoints = false;
   
     static const bool FirstPointIsCentroid = false;
   
     static const bool HasSelfChildren = false;
   
     static const bool RearrangesDataset = false;
   
     static const bool BinaryTree = false;
   
     static const bool UniqueNumDescendants = true;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
