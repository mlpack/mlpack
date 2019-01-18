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
 * Extra data for each node in the tree for the task of kernel density
 * estimation.
 */
class KDEStat
{
 public:
  //! Initialize the statistic.
  KDEStat() : validCentroid(false) { }

  //! Initialization for a fully initialized node.
  template<typename TreeType>
  KDEStat(TreeType& node)
  {
    // Calculate centroid if necessary.
    if (!tree::TreeTraits<TreeType>::FirstPointIsCentroid)
    {
      node.Center(centroid);
      validCentroid = true;
    }
    else
    {
      validCentroid = false;
    }
  }

  //! Get the centroid of the node.
  inline const arma::vec& Centroid() const
  {
    if (validCentroid)
      return centroid;
    throw std::logic_error("Centroid must be assigned before requesting its "
                           "value");
  }

  //! Modify the centroid of the node.
  void SetCentroid(arma::vec newCentroid)
  {
    validCentroid = true;
    centroid = std::move(newCentroid);
  }

  //! Get whether the centroid is valid.
  inline bool ValidCentroid() const { return validCentroid; }

  //! Serialize the statistic to/from an archive.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(centroid);
    ar & BOOST_SERIALIZATION_NVP(validCentroid);
  }

 private:
  //! Node centroid.
  arma::vec centroid;

  //! Whether the centroid is updated or is junk.
  bool validCentroid;
};

} // namespace kde
} // namespace mlpack

#endif
