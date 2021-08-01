
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_enumerate_tree.hpp:

Program Listing for File enumerate_tree.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_enumerate_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/enumerate_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_ENUMERATE_TREE_HPP
   #define MLPACK_CORE_TREE_ENUMERATE_TREE_HPP
   
   namespace mlpack {
   namespace tree  {
   namespace enumerate {
   
   // Actual implementation of the enumeration. The problem is the unified
   // detection if we're on the root, because Enter and Leave expect the
   // parent being passed.
   template <class TreeType, class Walker>
   void EnumerateTreeImpl(TreeType* tree, Walker& walker, bool root)
   {
     if (root)
       walker.Enter(tree, (const TreeType*)nullptr);
   
     const size_t numChildren = tree->NumChildren();
     for (size_t i = 0; i < numChildren; ++i)
     {
       TreeType* child = tree->ChildPtr(i);
       walker.Enter(child, tree);
       EnumerateTreeImpl(child, walker, false);
       walker.Leave(child, tree);
     }
   
     if (root)
       walker.Leave(tree, (const TreeType*)nullptr);
   }
   
   } // namespace enumerate
   
   
   template <class TreeType, class Walker>
   inline void EnumerateTree(TreeType* tree, Walker& walker)
   {
     enumerate::EnumerateTreeImpl(tree, walker, true);
   }
   
   } // namespace tree
   } // namespace mlpack
   
   
   #endif // MLPACK_CORE_TREE_ENUMERATE_TREE_HPP
