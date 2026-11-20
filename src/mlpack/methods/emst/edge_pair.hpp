/**
 * @file methods/emst/edge_pair.hpp
 *
 * @author Bill March (march@gatech.edu)
 *
 * This file contains utilities necessary for all of the minimum spanning tree
 * algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_EMST_EDGE_PAIR_HPP
#define MLPACK_METHODS_EMST_EDGE_PAIR_HPP

#include <mlpack/core/util/log.hpp>
#include <mlpack/prereqs.hpp>

#include "union_find.hpp"

namespace mlpack {

/**
 * An edge pair is simply two indices and a distance.  It is used as the
 * basic element of an edge list when computing a minimum spanning tree.
 */
class EdgePair
{
 private:
  //! Lesser index.
  size_t lesser;
  //! Greater index.
  size_t greater;
  //! Distance between two indices.
  double distance;

 public:
  /**
   * Initialize an EdgePair with two indices and a distance.  The indices are
   * called lesser and greater, implying that they be sorted before calling
   * Init.  However, this is not necessary for functionality; it is just a way
   * to keep the edge list organized in other code.
   */
  EdgePair(const size_t lesser, const size_t greater, const double dist) :
      lesser(lesser), greater(greater), distance(dist)
  {
    Log::Assert(lesser != greater,
        "EdgePair::EdgePair(): indices cannot be equal.");
  }

  //! Get the lesser index.
  size_t Lesser() const { return lesser; }
  //! Modify the lesser index.
  size_t& Lesser() { return lesser; }

  //! Get the greater index.
  size_t Greater() const { return greater; }
  //! Modify the greater index.
  size_t& Greater() { return greater; }

  //! Get the distance.
  double Distance() const { return distance; }
  //! Modify the distance.
  double& Distance() { return distance; }
}; // class EdgePair

} // namespace mlpack

#endif // MLPACK_METHODS_EMST_EDGE_PAIR_HPP
