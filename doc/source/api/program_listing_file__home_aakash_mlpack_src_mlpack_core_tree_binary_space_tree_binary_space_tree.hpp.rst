
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_binary_space_tree.hpp:

Program Listing for File binary_space_tree.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_binary_space_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/binary_space_tree/binary_space_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_BINARY_SPACE_TREE_HPP
   #define MLPACK_CORE_TREE_BINARY_SPACE_TREE_BINARY_SPACE_TREE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "../statistic.hpp"
   #include "midpoint_split.hpp"
   
   namespace mlpack {
   namespace tree  {
   
   template<typename MetricType,
            typename StatisticType = EmptyStatistic,
            typename MatType = arma::mat,
            template<typename BoundMetricType, typename...> class BoundType =
               bound::HRectBound,
            template<typename SplitBoundType, typename SplitMatType>
               class SplitType = MidpointSplit>
   class BinarySpaceTree
   {
    public:
     typedef MatType Mat;
     typedef typename MatType::elem_type ElemType;
   
     typedef SplitType<BoundType<MetricType>, MatType> Split;
   
    private:
     BinarySpaceTree* left;
     BinarySpaceTree* right;
     BinarySpaceTree* parent;
     size_t begin;
     size_t count;
     BoundType<MetricType> bound;
     StatisticType stat;
     ElemType parentDistance;
     ElemType furthestDescendantDistance;
     ElemType minimumBoundDistance;
     MatType* dataset;
   
    public:
     template<typename RuleType>
     class SingleTreeTraverser;
   
     template<typename RuleType>
     class DualTreeTraverser;
   
     template<typename RuleType>
     class BreadthFirstDualTreeTraverser;
   
     BinarySpaceTree(const MatType& data, const size_t maxLeafSize = 20);
   
     BinarySpaceTree(const MatType& data,
                     std::vector<size_t>& oldFromNew,
                     const size_t maxLeafSize = 20);
   
     BinarySpaceTree(const MatType& data,
                     std::vector<size_t>& oldFromNew,
                     std::vector<size_t>& newFromOld,
                     const size_t maxLeafSize = 20);
   
     BinarySpaceTree(MatType&& data,
                     const size_t maxLeafSize = 20);
   
     BinarySpaceTree(MatType&& data,
                     std::vector<size_t>& oldFromNew,
                     const size_t maxLeafSize = 20);
   
     BinarySpaceTree(MatType&& data,
                     std::vector<size_t>& oldFromNew,
                     std::vector<size_t>& newFromOld,
                     const size_t maxLeafSize = 20);
   
     BinarySpaceTree(BinarySpaceTree* parent,
                     const size_t begin,
                     const size_t count,
                     SplitType<BoundType<MetricType>, MatType>& splitter,
                     const size_t maxLeafSize = 20);
   
     BinarySpaceTree(BinarySpaceTree* parent,
                     const size_t begin,
                     const size_t count,
                     std::vector<size_t>& oldFromNew,
                     SplitType<BoundType<MetricType>, MatType>& splitter,
                     const size_t maxLeafSize = 20);
   
     BinarySpaceTree(BinarySpaceTree* parent,
                     const size_t begin,
                     const size_t count,
                     std::vector<size_t>& oldFromNew,
                     std::vector<size_t>& newFromOld,
                     SplitType<BoundType<MetricType>, MatType>& splitter,
                     const size_t maxLeafSize = 20);
   
     BinarySpaceTree(const BinarySpaceTree& other);
   
     BinarySpaceTree(BinarySpaceTree&& other);
   
     BinarySpaceTree& operator=(const BinarySpaceTree& other);
   
     BinarySpaceTree& operator=(BinarySpaceTree&& other);
   
     template<typename Archive>
     BinarySpaceTree(
         Archive& ar,
         const typename std::enable_if_t<cereal::is_loading<Archive>()>* = 0);
   
     ~BinarySpaceTree();
   
     const BoundType<MetricType>& Bound() const { return bound; }
     BoundType<MetricType>& Bound() { return bound; }
   
     const StatisticType& Stat() const { return stat; }
     StatisticType& Stat() { return stat; }
   
     bool IsLeaf() const;
   
     BinarySpaceTree* Left() const { return left; }
     BinarySpaceTree*& Left() { return left; }
   
     BinarySpaceTree* Right() const { return right; }
     BinarySpaceTree*& Right() { return right; }
   
     BinarySpaceTree* Parent() const { return parent; }
     BinarySpaceTree*& Parent() { return parent; }
   
     const MatType& Dataset() const { return *dataset; }
     MatType& Dataset() { return *dataset; }
   
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
   
     size_t GetNearestChild(const BinarySpaceTree& queryNode);
   
     size_t GetFurthestChild(const BinarySpaceTree& queryNode);
   
     ElemType FurthestPointDistance() const;
   
     ElemType FurthestDescendantDistance() const;
   
     ElemType MinimumBoundDistance() const;
   
     ElemType ParentDistance() const { return parentDistance; }
     ElemType& ParentDistance() { return parentDistance; }
   
     BinarySpaceTree& Child(const size_t child) const;
   
     BinarySpaceTree*& ChildPtr(const size_t child)
     { return (child == 0) ? left : right; }
   
     size_t NumPoints() const;
   
     size_t NumDescendants() const;
   
     size_t Descendant(const size_t index) const;
   
     size_t Point(const size_t index) const;
   
     ElemType MinDistance(const BinarySpaceTree& other) const
     {
       return bound.MinDistance(other.Bound());
     }
   
     ElemType MaxDistance(const BinarySpaceTree& other) const
     {
       return bound.MaxDistance(other.Bound());
     }
   
     math::RangeType<ElemType> RangeDistance(const BinarySpaceTree& other) const
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
   
     size_t Begin() const { return begin; }
     size_t& Begin() { return begin; }
   
     size_t Count() const { return count; }
     size_t& Count() { return count; }
   
     void Center(arma::vec& center) const { bound.Center(center); }
   
    private:
     void SplitNode(const size_t maxLeafSize,
                    SplitType<BoundType<MetricType>, MatType>& splitter);
   
     void SplitNode(std::vector<size_t>& oldFromNew,
                    const size_t maxLeafSize,
                    SplitType<BoundType<MetricType>, MatType>& splitter);
   
     template<typename BoundType2>
     void UpdateBound(BoundType2& boundToUpdate);
   
     void UpdateBound(bound::HollowBallBound<MetricType>& boundToUpdate);
   
    protected:
     BinarySpaceTree();
   
     friend class cereal::access;
   
    public:
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "binary_space_tree_impl.hpp"
   
   // Include everything else, if necessary.
   #include "../binary_space_tree.hpp"
   
   #endif
