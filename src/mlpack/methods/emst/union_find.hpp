/**
 * @file union_find.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Implements a union-find data structure.  This structure tracks the components
 * of a graph.  Each point in the graph is initially in its own component.
 * Calling unionfind.Union(x, y) unites the components indexed by x and y.
 * unionfind.Find(x) returns the index of the component containing point x.
 */
#ifndef __MLPACK_METHODS_EMST_UNION_FIND_HPP
#define __MLPACK_METHODS_EMST_UNION_FIND_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace emst {

/**
 * A Union-Find data structure.  See Cormen, Rivest, & Stein for details.
 */
class UnionFind
{
 private:
  arma::Col<size_t> parent_;
  arma::ivec rank_;
  size_t number_of_elements_;

 public:
  UnionFind() {}

  ~UnionFind() {}

  /**
   * Initializes the structure.  This implementation assumes
   * that the size is known advance and fixed
   *
   * @param size The number of elements to be tracked.
   */
  void Init(size_t size)
  {
    number_of_elements_ = size;
    parent_.set_size(number_of_elements_);
    rank_.set_size(number_of_elements_);
    for (size_t i = 0; i < number_of_elements_; i++)
    {
      parent_[i] = i;
      rank_[i] = 0;
    }
  }

  /**
   * Returns the component containing an element
   *
   * @param x the component to be found
   * @return The index of the component containing x
   */
  size_t Find(size_t x)
  {
    if (parent_[x] == x)
    {
      return x;
    }
    else
    {
      // This ensures that the tree has a small depth
      parent_[x] = Find(parent_[x]);
      return parent_[x];
    }
  }

  /**
   * @function Union
   *
   * Union the components containing x and y
   *
   * @param x one component
   * @param y the other component
   */
  void Union(size_t x, size_t y)
  {
    size_t x_root = Find(x);
    size_t y_root = Find(y);

    if (x_root == y_root)
    {
      return;
    }
    else if (rank_[x_root] == rank_[y_root])
    {
      parent_[y_root] = parent_[x_root];
      rank_[x_root] = rank_[x_root] + 1;
    }
    else if (rank_[x_root] > rank_[y_root])
    {
      parent_[y_root] = x_root;
    }
    else
    {
      parent_[x_root] = y_root;
    }
  }
}; // class UnionFind

}; // namespace emst
}; // namespace mlpack

#endif // __MLPACK_METHODS_EMST_UNION_FIND_HPP
