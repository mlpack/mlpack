
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_traversal_info.hpp:

Program Listing for File traversal_info.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_traversal_info.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/traversal_info.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_TRAVERSAL_INFO_HPP
   #define MLPACK_CORE_TREE_TRAVERSAL_INFO_HPP
   
   namespace mlpack {
   namespace tree {
   
   template<typename TreeType>
   class TraversalInfo
   {
    public:
     TraversalInfo() :
       lastQueryNode(NULL),
       lastReferenceNode(NULL),
       lastScore(0.0),
       lastBaseCase(0.0) { /* Nothing to do. */ }
   
     TreeType* LastQueryNode() const { return lastQueryNode; }
     TreeType*& LastQueryNode() { return lastQueryNode; }
   
     TreeType* LastReferenceNode() const { return lastReferenceNode; }
     TreeType*& LastReferenceNode() { return lastReferenceNode; }
   
     double LastScore() const { return lastScore; }
     double& LastScore() { return lastScore; }
   
     double LastBaseCase() const { return lastBaseCase; }
     double& LastBaseCase() { return lastBaseCase; }
   
    private:
     TreeType* lastQueryNode;
     TreeType* lastReferenceNode;
     double lastScore;
     double lastBaseCase;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
