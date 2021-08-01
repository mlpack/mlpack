
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_rann_ra_search_rules.hpp:

Program Listing for File ra_search_rules.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_rann_ra_search_rules.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/rann/ra_search_rules.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANN_RA_SEARCH_RULES_HPP
   #define MLPACK_METHODS_RANN_RA_SEARCH_RULES_HPP
   
   #include <mlpack/core/tree/traversal_info.hpp>
   
   #include <queue>
   
   namespace mlpack {
   namespace neighbor {
   
   template<typename SortPolicy, typename MetricType, typename TreeType>
   class RASearchRules
   {
    public:
     RASearchRules(const arma::mat& referenceSet,
                   const arma::mat& querySet,
                   const size_t k,
                   MetricType& metric,
                   const double tau = 5,
                   const double alpha = 0.95,
                   const bool naive = false,
                   const bool sampleAtLeaves = false,
                   const bool firstLeafExact = false,
                   const size_t singleSampleLimit = 20,
                   const bool sameSet = false);
   
     void GetResults(arma::Mat<size_t>& neighbors, arma::mat& distances);
   
     double BaseCase(const size_t queryIndex, const size_t referenceIndex);
   
     double Score(const size_t queryIndex, TreeType& referenceNode);
   
     double Score(const size_t queryIndex,
                  TreeType& referenceNode,
                  const double baseCaseResult);
   
     double Rescore(const size_t queryIndex,
                    TreeType& referenceNode,
                    const double oldScore);
   
     double Score(TreeType& queryNode, TreeType& referenceNode);
   
     double Score(TreeType& queryNode,
                  TreeType& referenceNode,
                  const double baseCaseResult);
   
     double Rescore(TreeType& queryNode,
                    TreeType& referenceNode,
                    const double oldScore);
   
   
     size_t NumDistComputations() { return numDistComputations; }
     size_t NumEffectiveSamples()
     {
       if (numSamplesMade.n_elem == 0)
         return 0;
       else
         return arma::sum(numSamplesMade);
     }
   
     typedef typename tree::TraversalInfo<TreeType> TraversalInfoType;
   
     const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
     TraversalInfoType& TraversalInfo() { return traversalInfo; }
   
     size_t MinimumBaseCases() const { return k; }
   
    private:
     const arma::mat& referenceSet;
   
     const arma::mat& querySet;
   
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
   
     bool sampleAtLeaves;
   
     bool firstLeafExact;
   
     size_t singleSampleLimit;
   
     size_t numSamplesReqd;
   
     arma::Col<size_t> numSamplesMade;
   
     double samplingRatio;
   
     size_t numDistComputations;
   
     bool sameSet;
   
     TraversalInfoType traversalInfo;
   
     void InsertNeighbor(const size_t queryIndex,
                         const size_t neighbor,
                         const double distance);
   
     double Score(const size_t queryIndex,
                  TreeType& referenceNode,
                  const double distance,
                  const double bestDistance);
   
     double Score(TreeType& queryNode,
                  TreeType& referenceNode,
                  const double distance,
                  const double bestDistance);
   
     static_assert(tree::TreeTraits<TreeType>::UniqueNumDescendants, "TreeType "
         "must provide a unique number of descendants points.");
   }; // class RASearchRules
   
   } // namespace neighbor
   } // namespace mlpack
   
   // Include implementation.
   #include "ra_search_rules_impl.hpp"
   
   #endif // MLPACK_METHODS_RANN_RA_SEARCH_RULES_HPP
