
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_sort_policies_nearest_neighbor_sort.hpp:

Program Listing for File nearest_neighbor_sort.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_sort_policies_nearest_neighbor_sort.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_SORT_HPP
   #define MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_SORT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace neighbor {
   
   class NearestNS
   {
    public:
     static inline bool IsBetter(const double value, const double ref)
     {
       return (value <= ref);
     }
   
     template<typename TreeType>
     static double BestNodeToNodeDistance(const TreeType* queryNode,
                                          const TreeType* referenceNode);
   
     template<typename TreeType>
     static double BestNodeToNodeDistance(const TreeType* queryNode,
                                          const TreeType* referenceNode,
                                          const double centerToCenterDistance);
   
     template<typename TreeType>
     static double BestNodeToNodeDistance(const TreeType* queryNode,
                                          const TreeType* referenceNode,
                                          const TreeType* referenceChildNode,
                                          const double centerToCenterDistance);
   
     template<typename VecType, typename TreeType>
     static double BestPointToNodeDistance(const VecType& queryPoint,
                                           const TreeType* referenceNode);
   
     template<typename VecType, typename TreeType>
     static double BestPointToNodeDistance(const VecType& queryPoint,
                                           const TreeType* referenceNode,
                                           const double pointToCenterDistance);
   
     template<typename VecType, typename TreeType>
     static size_t GetBestChild(const VecType& queryPoint, TreeType& referenceNode)
     {
       return referenceNode.GetNearestChild(queryPoint);
     };
   
     template<typename TreeType>
     static size_t GetBestChild(const TreeType& queryNode, TreeType& referenceNode)
     {
       return referenceNode.GetNearestChild(queryNode);
     };
   
     static inline double WorstDistance() { return DBL_MAX; }
   
     static inline double BestDistance() { return 0.0; }
   
     static inline double CombineBest(const double a, const double b)
     {
       return std::max(a - b, 0.0);
     }
   
     static inline double CombineWorst(const double a, const double b)
     {
       if (a == DBL_MAX || b == DBL_MAX)
         return DBL_MAX;
       return a + b;
     }
   
     static inline double Relax(const double value, const double epsilon)
     {
       if (value == DBL_MAX)
         return DBL_MAX;
       return (1 / (1 + epsilon)) * value;
     }
   
     static inline double ConvertToScore(const double distance)
     {
       return distance;
     }
   
     static inline double ConvertToDistance(const double score)
     {
       return score;
     }
   };
   
   // Due to an internal MinGW compiler bug (string table overflow) we have to
   // truncate the class name. For backward compatibility we setup an alias here.
   using NearestNeighborSort = NearestNS;
   
   } // namespace neighbor
   } // namespace mlpack
   
   // Include implementation of templated functions.
   #include "nearest_neighbor_sort_impl.hpp"
   
   #endif
