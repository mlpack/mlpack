
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_emst_dtb_rules.hpp:

Program Listing for File dtb_rules.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_emst_dtb_rules.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/emst/dtb_rules.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_EMST_DTB_RULES_HPP
   #define MLPACK_METHODS_EMST_DTB_RULES_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include <mlpack/core/tree/traversal_info.hpp>
   
   namespace mlpack {
   namespace emst {
   
   template<typename MetricType, typename TreeType>
   class DTBRules
   {
    public:
     DTBRules(const arma::mat& dataSet,
              UnionFind& connections,
              arma::vec& neighborsDistances,
              arma::Col<size_t>& neighborsInComponent,
              arma::Col<size_t>& neighborsOutComponent,
              MetricType& metric);
   
     double BaseCase(const size_t queryIndex, const size_t referenceIndex);
   
     double Score(const size_t queryIndex, TreeType& referenceNode);
   
     double Rescore(const size_t queryIndex,
                    TreeType& referenceNode,
                    const double oldScore);
   
     double Score(TreeType& queryNode, TreeType& referenceNode);
   
     double Rescore(TreeType& queryNode,
                    TreeType& referenceNode,
                    const double oldScore) const;
   
     typedef typename tree::TraversalInfo<TreeType> TraversalInfoType;
   
     const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
     TraversalInfoType& TraversalInfo() { return traversalInfo; }
   
     size_t BaseCases() const { return baseCases; }
     size_t& BaseCases() { return baseCases; }
   
     size_t Scores() const { return scores; }
     size_t& Scores() { return scores; }
   
    private:
     const arma::mat& dataSet;
   
     UnionFind& connections;
   
     arma::vec& neighborsDistances;
   
     arma::Col<size_t>& neighborsInComponent;
   
     arma::Col<size_t>& neighborsOutComponent;
   
     MetricType& metric;
   
     inline double CalculateBound(TreeType& queryNode) const;
   
     TraversalInfoType traversalInfo;
   
     size_t baseCases;
     size_t scores;
   }; // class DTBRules
   
   } // namespace emst
   } // namespace mlpack
   
   #include "dtb_rules_impl.hpp"
   
   #endif
