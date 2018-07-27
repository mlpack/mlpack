/**
 * @file kde_stat.hpp
 * @author Roberto Hueso
 *
 * Defines TreeStatType for KDE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KDE_STAT_HPP
#define MLPACK_METHODS_KDE_STAT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace kde {

/**
 * Extra data for each node in the tree.
 */
class KDEStat
{
 public:
  //! Initialize the statistic.
  KDEStat() { }

  //! Initialization for a fully initialized node.
  template<typename TreeType>
  KDEStat(TreeType& /* node */) { }

  //! Get the centroid calculation.
  const arma::vec& Centroid() const { return centroid; }

  //! Modify the centroid calculation.
  arma::vec& Centroid() { return centroid; }

  //! Serialize the statistic to/from an archive.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(centroid);
  }

 private:
  //! Node centroid.
  arma::vec centroid;
};

} // namespace kde
} // namespace mlpack

#endif
