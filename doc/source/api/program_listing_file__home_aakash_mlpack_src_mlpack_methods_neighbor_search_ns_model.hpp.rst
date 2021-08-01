
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_ns_model.hpp:

Program Listing for File ns_model.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_ns_model.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/ns_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_HPP
   #define MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_HPP
   
   #include <mlpack/core/tree/binary_space_tree.hpp>
   #include <mlpack/core/tree/cover_tree.hpp>
   #include <mlpack/core/tree/rectangle_tree.hpp>
   #include <mlpack/core/tree/spill_tree.hpp>
   #include <mlpack/core/tree/octree.hpp>
   #include "neighbor_search.hpp"
   
   namespace mlpack {
   namespace neighbor {
   
   class NSWrapperBase
   {
    public:
     NSWrapperBase() { }
   
     virtual NSWrapperBase* Clone() const = 0;
   
     virtual ~NSWrapperBase() { }
   
     virtual const arma::mat& Dataset() const = 0;
   
     virtual NeighborSearchMode SearchMode() const = 0;
     virtual NeighborSearchMode& SearchMode() = 0;
   
     virtual double Epsilon() const = 0;
     virtual double& Epsilon() = 0;
   
     virtual void Train(arma::mat&& referenceSet,
                        const size_t leafSize,
                        const double tau,
                        const double rho) = 0;
   
     virtual void Search(arma::mat&& querySet,
                         const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances,
                         const size_t leafSize,
                         const double rho) = 0;
   
     virtual void Search(const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances) = 0;
   };
   
   template<typename SortPolicy,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType,
            template<typename RuleType> class DualTreeTraversalType =
                TreeType<metric::EuclideanDistance,
                         NeighborSearchStat<SortPolicy>,
                         arma::mat>::template DualTreeTraverser,
            template<typename RuleType> class SingleTreeTraversalType =
                TreeType<metric::EuclideanDistance,
                         NeighborSearchStat<SortPolicy>,
                         arma::mat>::template SingleTreeTraverser>
   class NSWrapper : public NSWrapperBase
   {
    public:
     NSWrapper(const NeighborSearchMode searchMode,
               const double epsilon) :
         ns(searchMode, epsilon)
     {
       // Nothing else to do.
     }
   
     virtual ~NSWrapper() { }
   
     virtual NSWrapper* Clone() const { return new NSWrapper(*this); }
   
     const arma::mat& Dataset() const { return ns.ReferenceSet(); }
   
     NeighborSearchMode SearchMode() const { return ns.SearchMode(); }
     NeighborSearchMode& SearchMode() { return ns.SearchMode(); }
   
     double Epsilon() const { return ns.Epsilon(); }
     double& Epsilon() { return ns.Epsilon(); }
   
     virtual void Train(arma::mat&& referenceSet,
                        const size_t /* leafSize */,
                        const double /* tau */,
                        const double /* rho */);
   
     virtual void Search(arma::mat&& querySet,
                         const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances,
                         const size_t /* leafSize */,
                         const double /* rho */);
   
     virtual void Search(const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(ns));
     }
   
    protected:
     // Convenience typedef for the neighbor search type held by this class.
     typedef NeighborSearch<SortPolicy,
                            metric::EuclideanDistance,
                            arma::mat,
                            TreeType,
                            DualTreeTraversalType,
                            SingleTreeTraversalType> NSType;
   
     NSType ns;
   };
   
   template<typename SortPolicy,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType,
            template<typename RuleType> class DualTreeTraversalType =
                TreeType<metric::EuclideanDistance,
                         NeighborSearchStat<SortPolicy>,
                         arma::mat>::template DualTreeTraverser,
            template<typename RuleType> class SingleTreeTraversalType =
                TreeType<metric::EuclideanDistance,
                         NeighborSearchStat<SortPolicy>,
                         arma::mat>::template SingleTreeTraverser>
   class LeafSizeNSWrapper :
       public NSWrapper<SortPolicy,
                        TreeType,
                        DualTreeTraversalType,
                        SingleTreeTraversalType>
   {
    public:
     LeafSizeNSWrapper(const NeighborSearchMode searchMode,
                       const double epsilon) :
         NSWrapper<SortPolicy,
                   TreeType,
                   DualTreeTraversalType,
                   SingleTreeTraversalType>(searchMode, epsilon)
     {
       // Nothing to do.
     }
   
     virtual ~LeafSizeNSWrapper() { }
   
     virtual LeafSizeNSWrapper* Clone() const
     {
       return new LeafSizeNSWrapper(*this);
     }
   
     virtual void Train(arma::mat&& referenceSet,
                        const size_t leafSize,
                        const double /* tau */,
                        const double /* rho */);
   
     virtual void Search(arma::mat&& querySet,
                         const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances,
                         const size_t leafSize,
                         const double /* rho */);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(ns));
     }
   
    protected:
     using NSWrapper<SortPolicy,
                     TreeType,
                     DualTreeTraversalType,
                     SingleTreeTraversalType>::ns;
   };
   
   template<typename SortPolicy>
   class SpillNSWrapper :
       public NSWrapper<
           SortPolicy,
           tree::SPTree,
           tree::SPTree<metric::EuclideanDistance,
                        NeighborSearchStat<SortPolicy>,
                        arma::mat>::template DefeatistDualTreeTraverser,
           tree::SPTree<metric::EuclideanDistance,
                        NeighborSearchStat<SortPolicy>,
                        arma::mat>::template DefeatistSingleTreeTraverser>
   {
    public:
     SpillNSWrapper(const NeighborSearchMode searchMode,
                    const double epsilon) :
         NSWrapper<
             SortPolicy,
             tree::SPTree,
             tree::SPTree<metric::EuclideanDistance,
                          NeighborSearchStat<SortPolicy>,
                          arma::mat>::template DefeatistDualTreeTraverser,
             tree::SPTree<metric::EuclideanDistance,
                          NeighborSearchStat<SortPolicy>,
                          arma::mat>::template DefeatistSingleTreeTraverser>(
             searchMode, epsilon)
     {
       // Nothing to do.
     }
   
     virtual ~SpillNSWrapper() { }
   
     virtual SpillNSWrapper* Clone() const { return new SpillNSWrapper(*this); }
   
     virtual void Train(arma::mat&& referenceSet,
                        const size_t leafSize,
                        const double tau,
                        const double rho);
   
     virtual void Search(arma::mat&& querySet,
                         const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances,
                         const size_t leafSize,
                         const double rho);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(ns));
     }
   
    protected:
     using NSWrapper<
         SortPolicy,
         tree::SPTree,
         tree::SPTree<metric::EuclideanDistance,
                      NeighborSearchStat<SortPolicy>,
                      arma::mat>::template DefeatistDualTreeTraverser,
         tree::SPTree<metric::EuclideanDistance,
                      NeighborSearchStat<SortPolicy>,
                      arma::mat>::template DefeatistSingleTreeTraverser>::ns;
   };
   
   template<typename SortPolicy>
   class NSModel
   {
    public:
     enum TreeTypes
     {
       KD_TREE,
       COVER_TREE,
       R_TREE,
       R_STAR_TREE,
       BALL_TREE,
       X_TREE,
       HILBERT_R_TREE,
       R_PLUS_TREE,
       R_PLUS_PLUS_TREE,
       VP_TREE,
       RP_TREE,
       MAX_RP_TREE,
       SPILL_TREE,
       UB_TREE,
       OCTREE
     };
   
    private:
     TreeTypes treeType;
   
     bool randomBasis;
     arma::mat q;
   
     size_t leafSize;
     double tau;
     double rho;
   
     NSWrapperBase* nSearch;
   
    public:
     NSModel(TreeTypes treeType = TreeTypes::KD_TREE, bool randomBasis = false);
   
     NSModel(const NSModel& other);
   
     NSModel(NSModel&& other);
   
     NSModel& operator=(const NSModel& other);
   
     NSModel& operator=(NSModel&& other);
   
     ~NSModel();
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     const arma::mat& Dataset() const;
   
     NeighborSearchMode SearchMode() const;
     NeighborSearchMode& SearchMode();
   
     size_t LeafSize() const { return leafSize; }
     size_t& LeafSize() { return leafSize; }
   
     double Tau() const { return tau; }
     double& Tau() { return tau; }
   
     double Rho() const { return rho; }
     double& Rho() { return rho; }
   
     double Epsilon() const;
     double& Epsilon();
   
     TreeTypes TreeType() const { return treeType; }
     TreeTypes& TreeType() { return treeType; }
   
     bool RandomBasis() const { return randomBasis; }
     bool& RandomBasis() { return randomBasis; }
   
     void InitializeModel(const NeighborSearchMode searchMode,
                          const double epsilon);
   
     void BuildModel(arma::mat&& referenceSet,
                     const NeighborSearchMode searchMode,
                     const double epsilon = 0);
   
     void Search(arma::mat&& querySet,
                 const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     void Search(const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     std::string TreeName() const;
   };
   
   } // namespace neighbor
   } // namespace mlpack
   
   // Include implementation.
   #include "ns_model_impl.hpp"
   
   #endif
