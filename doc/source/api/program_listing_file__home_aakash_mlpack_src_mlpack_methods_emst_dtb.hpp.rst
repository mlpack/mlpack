
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_emst_dtb.hpp:

Program Listing for File dtb.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_emst_dtb.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/emst/dtb.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author = {March, W.B., Ram, P., and Gray, A.G.},
     title = {{Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis,
        Applications.}},
     booktitle = {Proceedings of the 16th ACM SIGKDD International Conference
        on Knowledge Discovery and Data Mining}
     series = {KDD 2010},
     year = {2010}
   }
   
   #ifndef MLPACK_METHODS_EMST_DTB_HPP
   #define MLPACK_METHODS_EMST_DTB_HPP
   
   #include "dtb_stat.hpp"
   #include "edge_pair.hpp"
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   
   #include <mlpack/core/tree/binary_space_tree.hpp>
   
   namespace mlpack {
   namespace emst  {
   
   template<
       typename MetricType = metric::EuclideanDistance,
       typename MatType = arma::mat,
       template<typename TreeMetricType,
                typename TreeStatType,
                typename TreeMatType> class TreeType = tree::KDTree
   >
   class DualTreeBoruvka
   {
    public:
     typedef TreeType<MetricType, DTBStat, MatType> Tree;
   
    private:
     std::vector<size_t> oldFromNew;
     Tree* tree;
     const MatType& data;
     bool ownTree;
   
     bool naive;
   
     std::vector<EdgePair> edges; // We must use vector with non-numerical types.
   
     UnionFind connections;
   
     arma::Col<size_t> neighborsInComponent;
     arma::Col<size_t> neighborsOutComponent;
     arma::vec neighborsDistances;
   
     double totalDist;
   
     MetricType metric;
   
     struct SortEdgesHelper
     {
       bool operator()(const EdgePair& pairA, const EdgePair& pairB)
       {
         return (pairA.Distance() < pairB.Distance());
       }
     } SortFun;
   
    public:
     DualTreeBoruvka(const MatType& dataset,
                     const bool naive = false,
                     const MetricType metric = MetricType());
   
     DualTreeBoruvka(Tree* tree,
                     const MetricType metric = MetricType());
   
     ~DualTreeBoruvka();
   
     void ComputeMST(arma::mat& results);
   
    private:
     void AddEdge(const size_t e1, const size_t e2, const double distance);
   
     void AddAllEdges();
   
     void EmitResults(arma::mat& results);
   
     void CleanupHelper(Tree* tree);
   
     void Cleanup();
   }; // class DualTreeBoruvka
   
   } // namespace emst
   } // namespace mlpack
   
   #include "dtb_impl.hpp"
   
   #endif // MLPACK_METHODS_EMST_DTB_HPP
