
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_fastmks_fastmks_rules.hpp:

Program Listing for File fastmks_rules.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_fastmks_fastmks_rules.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/fastmks/fastmks_rules.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_FASTMKS_FASTMKS_RULES_HPP
   #define MLPACK_METHODS_FASTMKS_FASTMKS_RULES_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/kernels/kernel_traits.hpp>
   #include <mlpack/core/tree/cover_tree/cover_tree.hpp>
   #include <mlpack/core/tree/traversal_info.hpp>
   #include <boost/heap/priority_queue.hpp>
   
   namespace mlpack {
   namespace fastmks {
   
   template<typename KernelType, typename TreeType>
   class FastMKSRules
   {
    public:
     FastMKSRules(const typename TreeType::Mat& referenceSet,
                  const typename TreeType::Mat& querySet,
                  const size_t k,
                  KernelType& kernel);
   
     void GetResults(arma::Mat<size_t>& indices, arma::mat& products);
   
     double BaseCase(const size_t queryIndex, const size_t referenceIndex);
   
     double Score(const size_t queryIndex, TreeType& referenceNode);
   
     double Score(TreeType& queryNode, TreeType& referenceNode);
   
     double Rescore(const size_t queryIndex,
                    TreeType& referenceNode,
                    const double oldScore) const;
   
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
   
    private:
     const typename TreeType::Mat& referenceSet;
     const typename TreeType::Mat& querySet;
   
     typedef std::pair<double, size_t> Candidate;
   
     struct CandidateCmp {
       bool operator()(const Candidate& c1, const Candidate& c2) const
       {
         return c1.first > c2.first;
       };
     };
   
     typedef boost::heap::priority_queue<Candidate,
         boost::heap::compare<CandidateCmp>> CandidateList;
   
     std::vector<CandidateList> candidates;
   
     const size_t k;
   
     arma::vec queryKernels;
     arma::vec referenceKernels;
   
     KernelType& kernel;
   
     size_t lastQueryIndex;
     size_t lastReferenceIndex;
     double lastKernel;
   
     double CalculateBound(TreeType& queryNode) const;
   
     void InsertNeighbor(const size_t queryIndex,
                         const size_t index,
                         const double product);
   
     size_t baseCases;
     size_t scores;
   
     TraversalInfoType traversalInfo;
   };
   
   } // namespace fastmks
   } // namespace mlpack
   
   // Include implementation.
   #include "fastmks_rules_impl.hpp"
   
   #endif
