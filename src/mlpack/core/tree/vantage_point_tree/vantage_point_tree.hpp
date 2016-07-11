/**
 * @file vantage_point_tree.hpp
 */

#ifndef MLPACK_CORE_TREE_VANTAGE_POINT_TREE_VANTAGE_POINT_TREE_HPP
#define MLPACK_CORE_TREE_VANTAGE_POINT_TREE_VANTAGE_POINT_TREE_HPP

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

template<typename MetricType,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat,
         template<typename BoundMetricType, typename...> class BoundType =
            bound::HRectBound,
         template<typename SplitBoundType, typename SplitMatType>
            class SplitType = MidpointSplit>
class VantagePointTree
{
 public:
  typedef MatType Mat;
  typedef typename MatType::elem_type ElemType;

 private:
  VantagePointTree* left;
  VantagePointTree* right;
  VantagePointTree* parent;
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

  VantagePointTree(const MatType& data, const size_t maxLeafSize = 20);

  VantagePointTree(const MatType& data,
                   std::vector<size_t>& oldFromNew,
                   const size_t maxLeafSize = 20);

  VantagePointTree(const MatType& data,
                   std::vector<size_t>& oldFromNew,
                   std::vector<size_t>& newFromOld,
                   const size_t maxLeafSize = 20);

  VantagePointTree(MatType&& data,
                   const size_t maxLeafSize = 20);

  VantagePointTree(MatType&& data,
                   std::vector<size_t>& oldFromNew,
                   const size_t maxLeafSize = 20);

  VantagePointTree(MatType&& data,
                   std::vector<size_t>& oldFromNew,
                   std::vector<size_t>& newFromOld,
                   const size_t maxLeafSize = 20);

  VantagePointTree(VantagePointTree* parent,
                   const size_t begin,
                   const size_t count,
                   SplitType<BoundType<MetricType>, MatType>& splitter,
                   const size_t maxLeafSize = 20);

  VantagePointTree(VantagePointTree* parent,
                   const size_t begin,
                   const size_t count,
                   std::vector<size_t>& oldFromNew,
                   SplitType<BoundType<MetricType>, MatType>& splitter,
                   const size_t maxLeafSize = 20);

  VantagePointTree(VantagePointTree* parent,
                   const size_t begin,
                   const size_t count,
                   std::vector<size_t>& oldFromNew,
                   std::vector<size_t>& newFromOld,
                   SplitType<BoundType<MetricType>, MatType>& splitter,
                   const size_t maxLeafSize = 20);

  VantagePointTree(const VantagePointTree& other);

  VantagePointTree(VantagePointTree&& other);

  template<typename Archive>
  VantagePointTree(
      Archive& ar,
      const typename boost::enable_if<typename Archive::is_loading>::type* = 0);

  ~VantagePointTree();

  const BoundType<MetricType>& Bound() const { return bound; }
  BoundType<MetricType>& Bound() { return bound; }

  const StatisticType& Stat() const { return stat; }
  StatisticType& Stat() { return stat; }

  bool IsLeaf() const;

  VantagePointTree* Left() const { return left; }
  VantagePointTree*& Left() { return left; }

  VantagePointTree* Right() const { return right; }
  VantagePointTree*& Right() { return right; }

  VantagePointTree* Parent() const { return parent; }
  VantagePointTree*& Parent() { return parent; }

  const MatType& Dataset() const { return *dataset; }
  MatType& Dataset() { return *dataset; }

  MetricType Metric() const { return MetricType(); }

  size_t NumChildren() const;

  ElemType FurthestPointDistance() const;

  ElemType FurthestDescendantDistance() const;

  ElemType MinimumBoundDistance() const;

  ElemType ParentDistance() const { return parentDistance; }
  ElemType& ParentDistance() { return parentDistance; }

  VantagePointTree& Child(const size_t child) const;

  VantagePointTree*& ChildPtr(const size_t child)
  { return (child == 0) ? left : right; }

  size_t NumPoints() const;

  size_t NumDescendants() const;

  size_t Descendant(const size_t index) const;

  size_t Point(const size_t index) const;

  ElemType MinDistance(const VantagePointTree* other) const
  {
    return bound.MinDistance(other->Bound());
  }

  ElemType MaxDistance(const VantagePointTree* other) const
  {
    return bound.MaxDistance(other->Bound());
  }

  math::RangeType<ElemType> RangeDistance(const VantagePointTree* other) const
  {
    return bound.RangeDistance(other->Bound());
  }

  template<typename VecType>
  ElemType MinDistance(const VecType& point,
                       typename boost::enable_if<IsVector<VecType> >::type* = 0)
      const
  {
    return bound.MinDistance(point);
  }

  template<typename VecType>
  ElemType MaxDistance(const VecType& point,
                       typename boost::enable_if<IsVector<VecType> >::type* = 0)
      const
  {
    return bound.MaxDistance(point);
  }

  template<typename VecType>
  math::RangeType<ElemType>
  RangeDistance(const VecType& point,
                typename boost::enable_if<IsVector<VecType> >::type* = 0) const
  {
    return bound.RangeDistance(point);
  }

  size_t Begin() const { return begin; }
  size_t& Begin() { return begin; }

  size_t Count() const { return count; }
  size_t& Count() { return count; }

  static bool HasSelfChildren() { return false; }

  void Center(arma::vec& center) { bound.Center(center); }

 private:
  void SplitNode(const size_t maxLeafSize,
                 SplitType<BoundType<MetricType>, MatType>& splitter);

  void SplitNode(std::vector<size_t>& oldFromNew,
                 const size_t maxLeafSize,
                 SplitType<BoundType<MetricType>, MatType>& splitter);

 protected:
  VantagePointTree();

  friend class boost::serialization::access;

 public:
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int version);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "vantage_point_tree_impl.hpp"

#endif // MLPACK_CORE_TREE_VANTAGE_POINT_TREE_VANTAGE_POINT_TREE_HPP
