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
#include <mlpack/core/tree/binary_space_tree.hpp>

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
  KDEStat() :
      validCentroid(false),
      mcBeta(0),
      mcAlpha(0),
      accumAlpha(0) { }

  //! Initialization for a fully initialized node.
  template<typename TreeType>
  KDEStat(TreeType& node) :
      mcBeta(0),
      mcAlpha(0),
      accumAlpha(0)
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

    // Build orthogonal base if possible.
    if (tree::TreeTraits<TreeType>::BinaryTree)
    {
      if (node.IsLeaf())
      {
        const size_t numPoints = node.NumPoints();
        const size_t firstPoint = node.Point(0);
        arma::mat pointsInTheNode =
            node.Dataset().cols(firstPoint, firstPoint + numPoints - 1);
        math::Center(pointsInTheNode, pointsInTheNode);

        // Calculate base mean.
        mean = arma::mean(pointsInTheNode, 1);

        // Right singular values. Unused.
        arma::mat v;

        // Singular value decomposition.
        if (pointsInTheNode.n_rows > pointsInTheNode.n_cols)
          arma::svd_econ(eigVec, eigVal, v, pointsInTheNode, "left");
        else
          arma::svd(eigVec, eigVal, v, pointsInTheNode);

        //eigVec.shed_cols(eigVec.n_cols - 3, eigVec.n_cols - 1);
        //eigVal.shed_rows(eigVal.n_rows - 3, eigVal.n_rows - 1);

        // Square singular values to get eigenvalues.
        eigVal %= eigVal / (pointsInTheNode.n_cols - 1);
      }
      else
      {
        const TreeType& child0 = node.Child(0);
        const TreeType& child1 = node.Child(1);
        const KDEStat& statChild0 = child0.Stat();
        const KDEStat& statChild1 = child1.Stat();

        if (child0.NumDescendants() == 0)
        {
          // Not used at the moment.
          eigVec = statChild1.eigVec;
          eigVal = statChild1.eigVal;
          mean = statChild1.mean;
        }
        else if (child1.NumDescendants() == 0)
        {
          // Not used at the moment.
          eigVec = statChild0.eigVec;
          eigVal = statChild0.eigVal;
          mean = statChild0.mean;
        }
        else
        {
          const arma::mat& U = statChild0.eigVec;
          const arma::mat& V = statChild1.eigVec;
          const arma::vec& mean0 = statChild0.mean;
          const arma::vec& mean1 = statChild1.mean;
          const arma::vec& eigVal0 = statChild0.eigVal;
          const arma::vec& eigVal1 = statChild1.eigVal;

          // Orthonormal basis.
          const arma::mat G = U.t() * V;
          arma::mat H = V - U * G;
          //arma::mat h = mean0 - U * U.t() * (mean0 - mean1);
          H.insert_cols(H.n_cols, mean0 - U * U.t() * (mean0 - mean1));
          const arma::mat v = arma::orth(H);

          // New eigenproblem.
          const size_t N = child0.NumDescendants();
          const size_t M = child1.NumDescendants();
          const size_t P = N + M;
          const size_t s = eigVal0.size() + v.n_cols;
          const arma::mat gamma = v.t() * V;
          arma::mat firstMat(s, s, arma::fill::zeros);
          firstMat.submat(0, 0, eigVal0.size() - 1, eigVal0.size() - 1) =
              arma::diagmat(eigVal0);

          arma::mat secondMat(s, s, arma::fill::zeros);
          // Upper left.
          secondMat.submat(0, 0, G.n_rows - 1, G.n_rows - 1) =
              G * arma::diagmat(eigVal1) * G.t();
          // Lower left.
          secondMat.submat(G.n_rows, 0, s - 1, G.n_rows - 1) =
              gamma * arma::diagmat(eigVal1) * G.t();
          // Upper right.
          secondMat.submat(0, G.n_rows, G.n_rows - 1, s - 1) =
              G * arma::diagmat(eigVal1) * gamma.t();
          // Lower right.
          secondMat.submat(G.n_rows, G.n_rows, s - 1, s - 1) =
              gamma * arma::diagmat(eigVal1) * gamma.t();

          arma::mat thirdMat(s, s, arma::fill::zeros);
          // Upper left.
          thirdMat.submat(0, 0, G.n_rows - 1, G.n_rows - 1) = G * G.t();
          // Lower left.
          thirdMat.submat(G.n_rows, 0, s - 1, G.n_rows - 1) = gamma * G.t();
          // Upper right.
          thirdMat.submat(0, G.n_rows, G.n_rows - 1, s - 1) = G * gamma.t();
          // Lower right.
          thirdMat.submat(G.n_rows, G.n_rows, s - 1, s - 1) = gamma * gamma.t();

          arma::mat newEigenProblem = ((double) N / P) * firstMat +
                                      ((double) M / P) * secondMat +
                                      ((double) N * M / (P * P)) * thirdMat;

          // Right singular values. Unused.
          arma::mat rightEigVal, newEigVec;
          arma::vec newEigVal;

          arma::svd(newEigVec, newEigVal, rightEigVal, newEigenProblem);

          arma::mat upsilon(U.n_rows, U.n_cols + v.n_cols);
          upsilon.cols(0, U.n_cols - 1) = U;
          upsilon.cols(U.n_cols, upsilon.n_cols - 1) = v;

          // Update node's orthogonal base.
          mean = (1 / (double) P) * (N * mean0 + M * mean1);
          eigVal = newEigVal;
          eigVec = upsilon * newEigVec;
        }
      }
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

  //! Get eigenvectors of the base.
  inline const arma::mat& EigVec() const { return eigVec; }

  //! Get eigenvalues of the base.
  inline const arma::vec& EigVal() const { return eigVal; }

  //! Get row mean of all descendant points of the node.
  inline const arma::vec& Mean() const { return mean; }

  //! Get accumulated Monte Carlo alpha of the node.
  inline double MCBeta() const { return mcBeta; }

  //! Modify accumulated Monte Carlo alpha of the node.
  inline double& MCBeta() { return mcBeta; }

  //! Get accumulated Monte Carlo alpha of the node.
  inline double AccumAlpha() const { return accumAlpha; }

  //! Modify accumulated Monte Carlo alpha of the node.
  inline double& AccumAlpha() { return accumAlpha; }

  //! Get Monte Carlo alpha of the node.
  inline double MCAlpha() const { return mcAlpha; }

  //! Modify Monte Carlo alpha of the node.
  inline double& MCAlpha() { return mcAlpha; }

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
  void serialize(Archive& ar, const unsigned int version)
  {
    ar & BOOST_SERIALIZATION_NVP(centroid);
    ar & BOOST_SERIALIZATION_NVP(validCentroid);

    // Backward compatibility: Old versions of KDEStat did not need to handle
    // alpha values.
    if (version > 0)
    {
      ar & BOOST_SERIALIZATION_NVP(mcBeta);
      ar & BOOST_SERIALIZATION_NVP(mcAlpha);
      ar & BOOST_SERIALIZATION_NVP(accumAlpha);
      ar & BOOST_SERIALIZATION_NVP(eigVec);
      ar & BOOST_SERIALIZATION_NVP(eigVal);
      ar & BOOST_SERIALIZATION_NVP(mean);
    }
    else if (Archive::is_loading::value)
    {
      mcBeta = -1;
      mcAlpha = -1;
      accumAlpha = -1;
      eigVec = arma::mat();
      eigVal = arma::vec();
      mean = arma::vec();
    }
  }

 private:
  //! Node centroid.
  arma::vec centroid;

  //! Whether the centroid is updated or is junk.
  bool validCentroid;

  //! Beta value for which mcAlpha is valid.
  double mcBeta;

  //! Monte Carlo alpha for this node.
  double mcAlpha;

  //! Accumulated not used Monte Carlo alpha in the current node.
  double accumAlpha;

  //! Eigenvectors of the base.
  arma::mat eigVec;

  //! Eigenvalues of the base.
  arma::vec eigVal;

  //! Row mean of all descendant points of the node.
  arma::vec mean;
};

} // namespace kde
} // namespace mlpack

//! Set the serialization version of the KDEStat class.
BOOST_TEMPLATE_CLASS_VERSION(template<>, mlpack::kde::KDEStat, 1);

#endif
