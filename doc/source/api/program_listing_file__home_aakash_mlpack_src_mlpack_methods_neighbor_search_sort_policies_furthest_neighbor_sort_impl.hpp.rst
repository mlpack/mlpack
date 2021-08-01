
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_sort_policies_furthest_neighbor_sort_impl.hpp:

Program Listing for File furthest_neighbor_sort_impl.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_sort_policies_furthest_neighbor_sort_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/sort_policies/furthest_neighbor_sort_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   /***
    * @file methods/neighbor_search/sort_policies/furthest_neighbor_sort_impl.hpp
    * @author Ryan Curtin
    *
    * Implementation of templated methods for the FurthestNeighborSort SortPolicy
    * class for the NeighborSearch class.
    *
    * mlpack is free software; you may redistribute it and/or modify it under the
    * terms of the 3-clause BSD license.  You should have received a copy of the
    * 3-clause BSD license along with mlpack.  If not, see
    * http://www.opensource.org/licenses/BSD-3-Clause for more information.
    */
   #ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_FURTHEST_NEIGHBOR_SORT_IMPL_HPP
   #define MLPACK_METHODS_NEIGHBOR_SEARCH_FURTHEST_NEIGHBOR_SORT_IMPL_HPP
   
   namespace mlpack {
   namespace neighbor {
   
   template<typename TreeType>
   inline double FurthestNS::BestNodeToNodeDistance(
       const TreeType* queryNode,
       const TreeType* referenceNode)
   {
     // This is not implemented yet for the general case because the trees do not
     // accept arbitrary distance metrics.
     return queryNode->MaxDistance(*referenceNode);
   }
   
   template<typename TreeType>
   inline double FurthestNS::BestNodeToNodeDistance(
       const TreeType* queryNode,
       const TreeType* referenceNode,
       const double centerToCenterDistance)
   {
     return queryNode->MaxDistance(*referenceNode, centerToCenterDistance);
   }
   
   template<typename TreeType>
   inline double FurthestNS::BestNodeToNodeDistance(
       const TreeType* queryNode,
       const TreeType* referenceNode,
       const TreeType* referenceChildNode,
       const double centerToCenterDistance)
   {
     return queryNode->MaxDistance(*referenceNode, centerToCenterDistance) +
         referenceChildNode->ParentDistance();
   }
   
   template<typename VecType, typename TreeType>
   inline double FurthestNS::BestPointToNodeDistance(
       const VecType& point,
       const TreeType* referenceNode)
   {
     // This is not implemented yet for the general case because the trees do not
     // accept arbitrary distance metrics.
     return referenceNode->MaxDistance(point);
   }
   
   template<typename VecType, typename TreeType>
   inline double FurthestNS::BestPointToNodeDistance(
       const VecType& point,
       const TreeType* referenceNode,
       const double pointToCenterDistance)
   {
     return referenceNode->MaxDistance(point, pointToCenterDistance);
   }
   
   } // namespace neighbor
   } // namespace mlpack
   
   #endif
