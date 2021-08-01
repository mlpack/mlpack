
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_spill_tree.hpp:

Program Listing for File spill_tree.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_spill_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/spill_tree/spill_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_SPILL_TREE_SPILL_TREE_HPP
   #define MLPACK_CORE_TREE_SPILL_TREE_SPILL_TREE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "../space_split/midpoint_space_split.hpp"
   #include "../statistic.hpp"
   
   namespace mlpack {
   namespace tree  {
   
   template<typename MetricType,
            typename StatisticType = EmptyStatistic,
            typename MatType = arma::mat,
            template<typename HyperplaneMetricType>
               class HyperplaneType = AxisOrthogonalHyperplane,
            template<typename SplitMetricType, typename SplitMatType>
               class SplitType = MidpointSpaceSplit>
   class SpillTree
   {
    public:
     typedef MatType Mat;
     typedef typename MatType::elem_type ElemType;
     typedef typename HyperplaneType<MetricType>::BoundType BoundType;
   
    private:
     SpillTree* left;
     SpillTree* right;
     SpillTree* parent;
     size_t count;
     arma::Col<size_t>* pointsIndex;
     bool overlappingNode;
     HyperplaneType<MetricType> hyperplane;
     BoundType bound;
     StatisticType stat;
     ElemType parentDistance;
     ElemType furthestDescendantDistance;
     ElemType minimumBoundDistance;
     const MatType* dataset;
     bool localDataset;
   
     template<typename RuleType, bool Defeatist = false>
     class SpillSingleTreeTraverser;
   
     template<typename RuleType, bool Defeatist = false>
     class SpillDualTreeTraverser;
   
    public:
     template<typename RuleType>
     using SingleTreeTraverser = SpillSingleTreeTraverser<RuleType, false>;
   
     template<typename RuleType>
     using DefeatistSingleTreeTraverser = SpillSingleTreeTraverser<RuleType, true>;
   
     template<typename RuleType>
     using DualTreeTraverser = SpillDualTreeTraverser<RuleType, false>;
   
     template<typename RuleType>
     using DefeatistDualTreeTraverser = SpillDualTreeTraverser<RuleType, true>;
   
     SpillTree(const MatType& data,
               const double tau = 0,
               const size_t maxLeafSize = 20,
               const double rho = 0.7);
   
     SpillTree(MatType&& data,
               const double tau = 0,
               const size_t maxLeafSize = 20,
               const double rho = 0.7);
   
     SpillTree(SpillTree* parent,
               arma::Col<size_t>& points,
               const double tau = 0,
               const size_t maxLeafSize = 20,
               const double rho = 0.7);
   
     SpillTree(const SpillTree& other);
   
     SpillTree(SpillTree&& other);
   
     SpillTree& operator=(const SpillTree& other);
   
     SpillTree& operator=(SpillTree&& other);
   
     template<typename Archive>
     SpillTree(
         Archive& ar,
         const typename std::enable_if_t<cereal::is_loading<Archive>()>* = 0);
   
     ~SpillTree();
   
     const BoundType& Bound() const { return bound; }
     BoundType& Bound() { return bound; }
   
     const StatisticType& Stat() const { return stat; }
     StatisticType& Stat() { return stat; }
   
     bool IsLeaf() const;
   
     SpillTree* Left() const { return left; }
     SpillTree*& Left() { return left; }
   
     SpillTree* Right() const { return right; }
     SpillTree*& Right() { return right; }
   
     SpillTree* Parent() const { return parent; }
     SpillTree*& Parent() { return parent; }
   
     const MatType& Dataset() const { return *dataset; }
   
     bool Overlap() const { return overlappingNode; }
   
     const HyperplaneType<MetricType>& Hyperplane() const { return hyperplane; }
   
     MetricType Metric() const { return MetricType(); }
   
     size_t NumChildren() const;
   
     template<typename VecType>
     size_t GetNearestChild(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0);
   
     template<typename VecType>
     size_t GetFurthestChild(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0);
   
     size_t GetNearestChild(const SpillTree& queryNode);
   
     size_t GetFurthestChild(const SpillTree& queryNode);
   
     ElemType FurthestPointDistance() const;
   
     ElemType FurthestDescendantDistance() const;
   
     ElemType MinimumBoundDistance() const;
   
     ElemType ParentDistance() const { return parentDistance; }
     ElemType& ParentDistance() { return parentDistance; }
   
     SpillTree& Child(const size_t child) const;
   
     SpillTree*& ChildPtr(const size_t child)
     { return (child == 0) ? left : right; }
   
     size_t NumPoints() const;
   
     size_t NumDescendants() const;
   
     size_t Descendant(const size_t index) const;
   
     size_t Point(const size_t index) const;
   
     ElemType MinDistance(const SpillTree& other) const
     {
       return bound.MinDistance(other.Bound());
     }
   
     ElemType MaxDistance(const SpillTree& other) const
     {
       return bound.MaxDistance(other.Bound());
     }
   
     math::RangeType<ElemType> RangeDistance(const SpillTree& other) const
     {
       return bound.RangeDistance(other.Bound());
     }
   
     template<typename VecType>
     ElemType MinDistance(const VecType& point,
                          typename std::enable_if_t<IsVector<VecType>::value>* = 0)
         const
     {
       return bound.MinDistance(point);
     }
   
     template<typename VecType>
     ElemType MaxDistance(const VecType& point,
                          typename std::enable_if_t<IsVector<VecType>::value>* = 0)
         const
     {
       return bound.MaxDistance(point);
     }
   
     template<typename VecType>
     math::RangeType<ElemType>
     RangeDistance(const VecType& point,
                   typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
     {
       return bound.RangeDistance(point);
     }
   
     static bool HasSelfChildren() { return false; }
   
     void Center(arma::vec& center) { bound.Center(center); }
   
    private:
     void SplitNode(arma::Col<size_t>& points,
                    const size_t maxLeafSize,
                    const double tau,
                    const double rho);
   
     bool SplitPoints(const double tau,
                      const double rho,
                      const arma::Col<size_t>& points,
                      arma::Col<size_t>& leftPoints,
                      arma::Col<size_t>& rightPoints);
    protected:
     SpillTree();
   
     friend class cereal::access;
   
    public:
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "spill_tree_impl.hpp"
   
   // Include everything else, if necessary.
   #include "../spill_tree.hpp"
   
   #endif
