/**
 * @file fastmks_stat.hpp
 * @author Ryan Curtin
 *
 * The statistic used in trees with FastMKS.
 *
 * This file is part of MLPACK 1.0.10.
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

  //! Get the last kernel evaluation.
  double LastKernel() const { return lastKernel; }
  //! Modify the last kernel evaluation.
  double& LastKernel() { return lastKernel; }

  //! Get the address of the node corresponding to the last distance evaluation.
  void* LastKernelNode() const { return lastKernelNode; }
  //! Modify the address of the node corresponding to the last distance
  //! evaluation.
  void*& LastKernelNode() { return lastKernelNode; }

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

}; // namespace fastmks
}; // namespace mlpack

#endif
