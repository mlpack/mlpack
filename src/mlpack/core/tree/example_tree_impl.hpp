/**
 * @file example_tree_impl.hpp
 * @author Ryan Curtin
 *
 * A fake implementation of the functions defined in ExampleTree.  Do not refer
 * to these as guidelines and do not use ExampleTree in your work because it
 * *will* *not* *work*.  The reason these implementations are here is so that we
 * can make tests that ensure mlpack dual-tree algorithms can compile with
 * *only* the functions specified in ExampleTree.
 */
#ifndef __MLPACK_CORE_TREE_EXAMPLE_TREE_IMPL_HPP
#define __MLPACK_CORE_TREE_EXAMPLE_TREE_IMPL_HPP

namespace mlpack {
namespace tree {

template<typename MetricType, typename StatisticType, typename MatType>
ExampleTree<MetricType, StatisticType, MatType>::ExampleTree(
    const MatType& dataset, MetricType& metric) : metric(metric), stat(*this)
{ }

template<typename MetricType, typename StatisticType, typename MatType>
size_t ExampleTree<MetricType, StatisticType, MatType>::NumChildren() const
{
  return 0;
}

template<typename MetricType, typename StatisticType, typename MatType>
const ExampleTree<MetricType, StatisticType, MatType>&
ExampleTree<MetricType, StatisticType, MatType>::Child(const size_t i) const
{
  return *this;
}

template<typename MetricType, typename StatisticType, typename MatType>
ExampleTree<MetricType, StatisticType, MatType>&
ExampleTree<MetricType, StatisticType, MatType>::Child(const size_t i)
{
  return *this;
}

template<typename MetricType, typename StatisticType, typename MatType>
ExampleTree<MetricType, StatisticType, MatType>*
ExampleTree<MetricType, StatisticType, MatType>::Parent() const
{
  return NULL;
}

template<typename MetricType, typename StatisticType, typename MatType>
size_t ExampleTree<MetricType, StatisticType, MatType>::NumPoints() const
{
  return 0;
}

template<typename MetricType, typename StatisticType, typename MatType>
size_t ExampleTree<MetricType, StatisticType, MatType>::Point(const size_t i)
    const
{
  return 0;
}

template<typename MetricType, typename StatisticType, typename MatType>
size_t ExampleTree<MetricType, StatisticType, MatType>::NumDescendants() const
{
  return 0;
}

template<typename MetricType, typename StatisticType, typename MatType>
size_t ExampleTree<MetricType, StatisticType, MatType>::Descendant(
    const size_t i) const
{
  return 0;
}

template<typename MetricType, typename StatisticType, typename MatType>
const StatisticType& ExampleTree<MetricType, StatisticType, MatType>::Stat()
    const
{
  return stat;
}

template<typename MetricType, typename StatisticType, typename MatType>
StatisticType& ExampleTree<MetricType, StatisticType, MatType>::Stat()
{
  return stat;
}

template<typename MetricType, typename StatisticType, typename MatType>
const MetricType& ExampleTree<MetricType, StatisticType, MatType>::Metric()
    const
{
  return metric;
}

template<typename MetricType, typename StatisticType, typename MatType>
MetricType& ExampleTree<MetricType, StatisticType, MatType>::Metric()
{
  return metric;
}

template<typename MetricType, typename StatisticType, typename MatType>
double ExampleTree<MetricType, StatisticType, MatType>::MinDistance(
    const MatType& point) const
{
  return 0.0;
}

template<typename MetricType, typename StatisticType, typename MatType>
double ExampleTree<MetricType, StatisticType, MatType>::MinDistance(
    const ExampleTree& node) const
{
  return 0.0;
}

template<typename MetricType, typename StatisticType, typename MatType>
double ExampleTree<MetricType, StatisticType, MatType>::MaxDistance(
    const MatType& point) const
{
  return 0.0;
}

template<typename MetricType, typename StatisticType, typename MatType>
double ExampleTree<MetricType, StatisticType, MatType>::MaxDistance(
    const ExampleTree& node) const
{
  return 0.0;
}

template<typename MetricType, typename StatisticType, typename MatType>
math::Range ExampleTree<MetricType, StatisticType, MatType>::RangeDistance(
    const MatType& point) const
{
  return math::Range(0.0, 0.0);
}

template<typename MetricType, typename StatisticType, typename MatType>
math::Range ExampleTree<MetricType, StatisticType, MatType>::RangeDistance(
    const ExampleTree& node) const
{
  return math::Range(0.0, 0.0);
}

template<typename MetricType, typename StatisticType, typename MatType>
void ExampleTree<MetricType, StatisticType, MatType>::Centroid(
    arma::vec& centroid) const
{ }

template<typename MetricType, typename StatisticType, typename MatType>
double ExampleTree<MetricType, StatisticType, MatType>::
    FurthestDescendantDistance() const
{
  return 0.0;
}

template<typename MetricType, typename StatisticType, typename MatType>
double ExampleTree<MetricType, StatisticType, MatType>::ParentDistance() const
{
  return 0.0;
}

}; // namespace tree
}; // namespace mlpack

#endif
