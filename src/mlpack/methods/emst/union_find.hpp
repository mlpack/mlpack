/**
 * @file union_find.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Implements a union-find data structure.  This structure tracks the components
 * of a graph.  Each point in the graph is initially in its own component.
 * Calling unionfind.Union(x, y) unites the components indexed by x and y.
 * unionfind.Find(x) returns the index of the component containing point x.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_EMST_UNION_FIND_HPP
#define __MLPACK_METHODS_EMST_UNION_FIND_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace emst {

/**
 * A Union-Find data structure.  See Cormen, Rivest, & Stein for details.  The
 * structure tracks the components of a graph.  Each point in the graph is
 * initially in its own component.  Calling Union(x, y) unites the components
 * indexed by x and y.  Find(x) returns the index of the component containing
 * point x.
 */
class UnionFind
{
 private:
  arma::Col<size_t> parent;
  arma::ivec rank;

 public:
  //! Construct the object with the given size.
  UnionFind(const size_t size) : parent(size), rank(size)
  {
    for (size_t i = 0; i < size; ++i)
    {
      parent[i] = i;
      rank[i] = 0;
    }
  }

  //! Destroy the object (nothing to do).
  ~UnionFind() { }

  /**
   * Returns the component containing an element.
   *
   * @param x the component to be found
   * @return The index of the component containing x
   */
  size_t Find(const size_t x)
  {
    if (parent[x] == x)
    {
      return x;
    }
    else
    {
      // This ensures that the tree has a small depth
      parent[x] = Find(parent[x]);
      return parent[x];
    }
  }

  /**
   * Union the components containing x and y.
   *
   * @param x one component
   * @param y the other component
   */
  void Union(const size_t x, const size_t y)
  {
    const size_t xRoot = Find(x);
    const size_t yRoot = Find(y);

    if (xRoot == yRoot)
    {
      return;
    }
    else if (rank[xRoot] == rank[yRoot])
    {
      parent[yRoot] = parent[xRoot];
      rank[xRoot] = rank[xRoot] + 1;
    }
    else if (rank[xRoot] > rank[yRoot])
    {
      parent[yRoot] = xRoot;
    }
    else
    {
      parent[xRoot] = yRoot;
    }
  }
}; // class UnionFind

}; // namespace emst
}; // namespace mlpack

#endif // __MLPACK_METHODS_EMST_UNION_FIND_HPP
