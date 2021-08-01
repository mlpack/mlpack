
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_range_search_range_search_rules.hpp:

Program Listing for File range_search_rules.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_range_search_range_search_rules.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/range_search/range_search_rules.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_HPP
   #define MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_HPP
   
   #include <mlpack/core/tree/traversal_info.hpp>
   
   namespace mlpack {
   namespace range {
   
   template<typename MetricType, typename TreeType>
   class RangeSearchRules
   {
    public:
     RangeSearchRules(const arma::mat& referenceSet,
                      const arma::mat& querySet,
                      const math::Range& range,
                      std::vector<std::vector<size_t> >& neighbors,
                      std::vector<std::vector<double> >& distances,
                      MetricType& metric,
                      const bool sameSet = false);
   
     double BaseCase(const size_t queryIndex, const size_t referenceIndex);
   
     double Score(const size_t queryIndex, TreeType& referenceNode);
   
     double Rescore(const size_t queryIndex,
                    TreeType& referenceNode,
                    const double oldScore) const;
   
     double Score(TreeType& queryNode, TreeType& referenceNode);
   
     double Rescore(TreeType& queryNode,
                    TreeType& referenceNode,
                    const double oldScore) const;
   
     typedef typename tree::TraversalInfo<TreeType> TraversalInfoType;
   
     const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
     TraversalInfoType& TraversalInfo() { return traversalInfo; }
   
     size_t BaseCases() const { return baseCases; }
     size_t Scores() const { return scores; }
   
     size_t MinimumBaseCases() const { return 0; }
   
    private:
     const arma::mat& referenceSet;
   
     const arma::mat& querySet;
   
     const math::Range& range;
   
     std::vector<std::vector<size_t> >& neighbors;
   
     std::vector<std::vector<double> >& distances;
   
     MetricType& metric;
   
     bool sameSet;
   
     size_t lastQueryIndex;
     size_t lastReferenceIndex;
   
     void AddResult(const size_t queryIndex,
                    TreeType& referenceNode);
   
     TraversalInfoType traversalInfo;
   
     size_t baseCases;
     size_t scores;
   };
   
   } // namespace range
   } // namespace mlpack
   
   // Include implementation.
   #include "range_search_rules_impl.hpp"
   
   #endif
