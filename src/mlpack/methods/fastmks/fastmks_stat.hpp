/**
 * @file fastmks_stat.hpp
 * @author Ryan Curtin
 *
 * The statistic used in trees with FastMKS.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_FASTMKS_FASTMKS_STAT_HPP
#define MLPACK_METHODS_FASTMKS_FASTMKS_STAT_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/tree_traits.hpp>

namespace mlpack {
namespace fastmks {

/**
 * The statistic used in trees with FastMKS.  This stores both the bound and the
 * self-kernels for each node in the tree.
 */
class FastMKSStat
{
 public:
  /**
   * Default initialization.
   */
  FastMKSStat() :
      bound(-DBL_MAX),
      selfKernel(0.0),
      lastKernel(0.0),
      lastKernelNode(NULL)
  { }

  /**
   * Initialize this statistic for the given tree node.  The TreeType's metric
   * better be IPMetric with some kernel type (that is, Metric().Kernel() must
   * exist).
   *
   * @param node Node that this statistic is built for.
   */
  template<typename TreeType>
  FastMKSStat(const TreeType& node) :
      bound(-DBL_MAX),
      lastKernel(0.0),
      lastKernelNode(NULL)
  {
    // Do we have to calculate the centroid?
    if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
    {
      // If this type of tree has self-children, then maybe the evaluation is
      // already done.  These statistics are built bottom-up, so the child stat
      // should already be done.
      if ((tree::TreeTraits<TreeType>::HasSelfChildren) &&
          (node.NumChildren() > 0) &&
          (node.Point(0) == node.Child(0).Point(0)))
      {
        selfKernel = node.Child(0).Stat().SelfKernel();
      }
      else
      {
        selfKernel = sqrt(node.Metric().Kernel().Evaluate(
            node.Dataset().col(node.Point(0)),
            node.Dataset().col(node.Point(0))));
      }
    }
    else
    {
      // Calculate the centroid.
      arma::vec center;
      node.Center(center);

      selfKernel = sqrt(node.Metric().Kernel().Evaluate(center, center));
    }
  }

  //! Get the self-kernel.
  double SelfKernel() const { return selfKernel; }
  //! Modify the self-kernel.
  double& SelfKernel() { return selfKernel; }

  //! Get the bound.
  double Bound() const { return bound; }
  //! Modify the bound.
  double& Bound() { return bound; }

  //! Get the last kernel evaluation.
  double LastKernel() const { return lastKernel; }
  //! Modify the last kernel evaluation.
  double& LastKernel() { return lastKernel; }

  //! Get the address of the node corresponding to the last distance evaluation.
  void* LastKernelNode() const { return lastKernelNode; }
  //! Modify the address of the node corresponding to the last distance
  //! evaluation.
  void*& LastKernelNode() { return lastKernelNode; }

  //! Serialize the statistic.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(bound, "bound");
    ar & data::CreateNVP(selfKernel, "selfKernel");

    // Void out last kernel information on load.
    if (Archive::is_loading::value)
    {
      lastKernel = 0.0;
      lastKernelNode = NULL;
    }
  }

 private:
  //! The bound for pruning.
  double bound;

  //! The self-kernel evaluation: sqrt(K(centroid, centroid)).
  double selfKernel;

  //! The last kernel evaluation.
  double lastKernel;

  //! The node corresponding to the last kernel evaluation.  This has to be void
  //! otherwise we get recursive template arguments.
  void* lastKernelNode;
};

} // namespace fastmks
} // namespace mlpack

#endif
