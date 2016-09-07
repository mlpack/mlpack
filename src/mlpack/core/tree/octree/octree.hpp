/**
 * @file octree.hpp
 * @author Ryan Curtin
 *
 * Definition of generalized octree (Octree).
 */
#ifndef MLPACK_CORE_TREE_OCTREE_OCTREE_HPP
#define MLPACK_CORE_TREE_OCTREE_OCTREE_HPP

#include <mlpack/core.hpp>
#include "../hrectbound.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat>
class Octree
{
 public:
  //! So other classes can use TreeType::Mat.
  typedef MatType Mat;
  //! The type of element held in MatType.
  typedef typename MatType::elem_type ElemType;

 private:
  //! The children held by this node.
  std::vector<Octree*> children;

  //! The index of the first point in the dataset contained in this node (and
  //! its children).
  size_t begin;
  //! The number of points of the dataset contained in this node (and its
  //! children).
  size_t count;

  
};
