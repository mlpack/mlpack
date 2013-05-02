/**
 * @file fastmks_stat.hpp
 * @author Ryan Curtin
 *
 * The statistic used in trees with FastMKS.
 */
#ifndef __MLPACK_METHODS_FASTMKS_FASTMKS_STAT_HPP
#define __MLPACK_METHODS_FASTMKS_FASTMKS_STAT_HPP

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
  FastMKSStat() : bound(-DBL_MAX), selfKernel(0.0) { }

  /**
   * Initialize this statistic for the given tree node.  The TreeType's metric
   * better be IPMetric with some kernel type (that is, Metric().Kernel() must
   * exist).
   *
   * @param node Node that this statistic is built for.
   */
  template<typename TreeType>
  FastMKSStat(const TreeType& node) :
      bound(-DBL_MAX)
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
            node.Dataset().unsafe_col(node.Point(0)),
            node.Dataset().unsafe_col(node.Point(0))));
      }
    }
    else
    {
      // Calculate the centroid.
      arma::vec centroid;
      node.Centroid(centroid);

      selfKernel = sqrt(node.Metric().Kernel().Evaluate(centroid, centroid));
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

 private:
  //! The bound for pruning.
  double bound;

  //! The self-kernel evaluation: sqrt(K(centroid, centroid)).
  double selfKernel;
};

}; // namespace fastmks
}; // namespace mlpack

#endif
