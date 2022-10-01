/**
 * @file methods/kde/kde_stat.hpp
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

/**
 * Extra data for each node in the tree for the task of kernel density
 * estimation.
 */
class KDEStat
{
 public:
  //! Initialize the statistic.
  KDEStat() :
      mcBeta(0),
      mcAlpha(0),
      accumAlpha(0),
      accumError(0)
  { /* Nothing to do.*/ }

  //! Initialization for a fully initialized node.
  template<typename TreeType>
  KDEStat(TreeType& /* node */) :
      mcBeta(0),
      mcAlpha(0),
      accumAlpha(0),
      accumError(0)
  { /* Nothing to do. */ }

  //! Get accumulated Monte Carlo alpha of the node.
  inline double MCBeta() const { return mcBeta; }

  //! Modify accumulated Monte Carlo alpha of the node.
  inline double& MCBeta() { return mcBeta; }

  //! Get accumulated Monte Carlo alpha of the node.
  inline double AccumAlpha() const { return accumAlpha; }

  //! Modify accumulated Monte Carlo alpha of the node.
  inline double& AccumAlpha() { return accumAlpha; }

  //! Get accumulated error tolerance of the node.
  inline double AccumError() const { return accumError; }

  //! Modify accumulated error tolerance of the node.
  inline double& AccumError() { return accumError; }

  //! Get Monte Carlo alpha of the node.
  inline double MCAlpha() const { return mcAlpha; }

  //! Modify Monte Carlo alpha of the node.
  inline double& MCAlpha() { return mcAlpha; }

  //! Serialize the statistic to/from an archive.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(mcBeta));
    ar(CEREAL_NVP(mcAlpha));
    ar(CEREAL_NVP(accumAlpha));
    ar(CEREAL_NVP(accumError));
  }

 private:
  //! Beta value for which mcAlpha is valid.
  double mcBeta;

  //! Monte Carlo alpha for this node.
  double mcAlpha;

  //! Accumulated not used Monte Carlo alpha in the current node.
  double accumAlpha;

  //! Accumulated not used error tolerance in the current node.
  double accumError;
};

} // namespace mlpack

#endif
