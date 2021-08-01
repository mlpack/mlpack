
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_fastmks_fastmks.hpp:

Program Listing for File fastmks.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_fastmks_fastmks.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/fastmks/fastmks.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_FASTMKS_FASTMKS_HPP
   #define MLPACK_METHODS_FASTMKS_FASTMKS_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/ip_metric.hpp>
   #include "fastmks_stat.hpp"
   #include <mlpack/core/tree/cover_tree.hpp>
   #include <queue>
   
   namespace mlpack {
   namespace fastmks  {
   
   template<
       typename KernelType,
       typename MatType = arma::mat,
       template<typename TreeMetricType,
                typename TreeStatType,
                typename TreeMatType> class TreeType = tree::StandardCoverTree
   >
   class FastMKS
   {
    public:
     typedef TreeType<metric::IPMetric<KernelType>, FastMKSStat, MatType> Tree;
   
     FastMKS(const bool singleMode = false, const bool naive = false);
   
     FastMKS(const MatType& referenceSet,
             const bool singleMode = false,
             const bool naive = false);
   
     FastMKS(const MatType& referenceSet,
             KernelType& kernel,
             const bool singleMode = false,
             const bool naive = false);
   
     FastMKS(MatType&& referenceSet,
             const bool singleMode = false,
             const bool naive = false);
   
     FastMKS(MatType&& referenceSet,
             KernelType& kernel,
             const bool singleMode = false,
             const bool naive = false);
   
     FastMKS(Tree* referenceTree,
             const bool singleMode = false);
   
     FastMKS(const FastMKS& other);
   
     FastMKS(FastMKS&& other);
   
     FastMKS& operator=(const FastMKS& other);
   
     FastMKS& operator=(FastMKS&& other);
   
     ~FastMKS();
   
     void Train(const MatType& referenceSet);
   
     void Train(const MatType& referenceSet, KernelType& kernel);
   
     void Train(MatType&& referenceSet);
   
     void Train(MatType&& referenceSet, KernelType& kernel);
   
     void Train(Tree* referenceTree);
   
     void Search(const MatType& querySet,
                 const size_t k,
                 arma::Mat<size_t>& indices,
                 arma::mat& kernels);
   
     void Search(Tree* querySet,
                 const size_t k,
                 arma::Mat<size_t>& indices,
                 arma::mat& kernels);
   
     void Search(const size_t k,
                 arma::Mat<size_t>& indices,
                 arma::mat& products);
   
     const metric::IPMetric<KernelType>& Metric() const { return metric; }
     metric::IPMetric<KernelType>& Metric() { return metric; }
   
     bool SingleMode() const { return singleMode; }
     bool& SingleMode() { return singleMode; }
   
     bool Naive() const { return naive; }
     bool& Naive() { return naive; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     const MatType* referenceSet;
     Tree* referenceTree;
     bool treeOwner;
     bool setOwner;
   
     bool singleMode;
     bool naive;
   
     metric::IPMetric<KernelType> metric;
   
     typedef std::pair<double, size_t> Candidate;
   
     struct CandidateCmp {
       bool operator()(const Candidate& c1, const Candidate& c2)
       {
         return c1.first > c2.first;
       };
     };
   
     typedef std::priority_queue<Candidate, std::vector<Candidate>,
         CandidateCmp> CandidateList;
   };
   
   } // namespace fastmks
   } // namespace mlpack
   
   // Include implementation.
   #include "fastmks_impl.hpp"
   
   #endif
