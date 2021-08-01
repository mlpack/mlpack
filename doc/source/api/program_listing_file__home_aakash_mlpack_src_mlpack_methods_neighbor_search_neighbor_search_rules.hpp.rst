
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_neighbor_search_rules.hpp:

Program Listing for File neighbor_search_rules.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_neighbor_search_rules.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/neighbor_search_rules.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_RULES_HPP
   #define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_RULES_HPP
   
   #include <mlpack/core/tree/traversal_info.hpp>
   
   #include <queue>
   
   namespace mlpack {
   namespace neighbor {
   
   template<typename SortPolicy, typename MetricType, typename TreeType>
   class NeighborSearchRules
   {
    public:
     NeighborSearchRules(const typename TreeType::Mat& referenceSet,
                         const typename TreeType::Mat& querySet,
                         const size_t k,
                         MetricType& metric,
                         const double epsilon = 0,
                         const bool sameSet = false);
   
     void GetResults(arma::Mat<size_t>& neighbors, arma::mat& distances);
   
     double BaseCase(const size_t queryIndex, const size_t referenceIndex);
   
     double Score(const size_t queryIndex, TreeType& referenceNode);
   
     size_t GetBestChild(const size_t queryIndex, TreeType& referenceNode);
   
     size_t GetBestChild(const TreeType& queryNode, TreeType& referenceNode);
   
     double Rescore(const size_t queryIndex,
                    TreeType& referenceNode,
                    const double oldScore) const;
   
     double Score(TreeType& queryNode, TreeType& referenceNode);
   
     double Rescore(TreeType& queryNode,
                    TreeType& referenceNode,
                    const double oldScore) const;
   
     size_t BaseCases() const { return baseCases; }
     size_t& BaseCases() { return baseCases; }
   
     size_t Scores() const { return scores; }
     size_t& Scores() { return scores; }
   
     typedef typename tree::TraversalInfo<TreeType> TraversalInfoType;
   
     const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
     TraversalInfoType& TraversalInfo() { return traversalInfo; }
   
     size_t MinimumBaseCases() const { return k; }
   
    protected:
     const typename TreeType::Mat& referenceSet;
   
     const typename TreeType::Mat& querySet;
   
     typedef std::pair<double, size_t> Candidate;
   
     struct CandidateCmp {
       bool operator()(const Candidate& c1, const Candidate& c2)
       {
         return !SortPolicy::IsBetter(c2.first, c1.first);
       };
     };
   
     typedef std::priority_queue<Candidate, std::vector<Candidate>, CandidateCmp>
         CandidateList;
   
     std::vector<CandidateList> candidates;
   
     const size_t k;
   
     MetricType& metric;
   
     bool sameSet;
   
     const double epsilon;
   
     size_t lastQueryIndex;
     size_t lastReferenceIndex;
     double lastBaseCase;
   
     size_t baseCases;
     size_t scores;
   
     TraversalInfoType traversalInfo;
   
     double CalculateBound(TreeType& queryNode) const;
   
     void InsertNeighbor(const size_t queryIndex,
                         const size_t neighbor,
                         const double distance);
   };
   
   } // namespace neighbor
   } // namespace mlpack
   
   // Include implementation.
   #include "neighbor_search_rules_impl.hpp"
   
   #endif // MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_RULES_HPP
