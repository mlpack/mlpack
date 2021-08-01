
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_bound_traits.hpp:

Program Listing for File bound_traits.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_bound_traits.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/bound_traits.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BOUND_TRAITS_HPP
   #define MLPACK_CORE_TREE_BOUND_TRAITS_HPP
   
   namespace mlpack {
   namespace bound {
   
   template<typename BoundType>
   struct BoundTraits
   {
     static const bool HasTightBounds = false;
   };
   
   } // namespace bound
   } // namespace mlpack
   
   #endif
