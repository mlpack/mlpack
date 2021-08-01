
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_sort_policies_nearest_neighbor_sort_impl.hpp:

Program Listing for File nearest_neighbor_sort_impl.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_sort_policies_nearest_neighbor_sort_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_NEIGHBOR_NEAREST_NEIGHBOR_SORT_IMPL_HPP
   #define MLPACK_NEIGHBOR_NEAREST_NEIGHBOR_SORT_IMPL_HPP
   
   namespace mlpack {
   namespace neighbor {
   
   template<typename TreeType>
   inline double NearestNS::BestNodeToNodeDistance(
       const TreeType* queryNode,
       const TreeType* referenceNode)
   {
     // This is not implemented yet for the general case because the trees do not
     // accept arbitrary distance metrics.
     return queryNode->MinDistance(*referenceNode);
   }
   
   template<typename TreeType>
   inline double NearestNS::BestNodeToNodeDistance(
       const TreeType* queryNode,
       const TreeType* referenceNode,
       const double centerToCenterDistance)
   {
     return queryNode->MinDistance(*referenceNode, centerToCenterDistance);
   }
   
   template<typename TreeType>
   inline double NearestNS::BestNodeToNodeDistance(
       const TreeType* queryNode,
       const TreeType* /* referenceNode */,
       const TreeType* referenceChildNode,
       const double centerToCenterDistance)
   {
     return queryNode->MinDistance(*referenceChildNode, centerToCenterDistance) -
         referenceChildNode->ParentDistance();
   }
   
   template<typename VecType, typename TreeType>
   inline double NearestNS::BestPointToNodeDistance(
       const VecType& point,
       const TreeType* referenceNode)
   {
     // This is not implemented yet for the general case because the trees do not
     // accept arbitrary distance metrics.
     return referenceNode->MinDistance(point);
   }
   
   template<typename VecType, typename TreeType>
   inline double NearestNS::BestPointToNodeDistance(
       const VecType& point,
       const TreeType* referenceNode,
       const double pointToCenterDistance)
   {
     return referenceNode->MinDistance(point, pointToCenterDistance);
   }
   
   } // namespace neighbor
   } // namespace mlpack
   
   #endif
