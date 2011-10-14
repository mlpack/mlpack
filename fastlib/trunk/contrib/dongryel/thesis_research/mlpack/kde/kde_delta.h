/** @file kde_delta.h
 *
 *  The delta quantities that are used in making the pruning decisions
 *  in kde dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_DELTA_H
#define MLPACK_KDE_KDE_DELTA_H

namespace mlpack {
namespace kde {

class KdeDelta {

  public:

    double densities_l_;

    double densities_u_;

    unsigned long int pruned_;

    double used_error_;

    int order_farfield_to_local_;

    int order_farfield_;

    int order_local_;

    std::vector< core::monte_carlo::MeanVariancePair > *mean_variance_pair_;

    KdeDelta() {
      SetZero();
    }

    void SetZero() {
      densities_l_ = densities_u_ = used_error_ = 0;
      pruned_ = 0;
      order_farfield_to_local_ = -1;
      order_farfield_ = -1;
      order_local_ = -1;
      mean_variance_pair_ = NULL;
    }

    template<typename MetricType, typename GlobalType, typename TreeType>
    void DeterministicCompute(
      const MetricType &metric,
      const GlobalType &global, TreeType *qnode, TreeType *rnode,
      bool qnode_and_rnode_are_equal,
      const core::math::Range &squared_distance_range) {

      int rnode_count = (global.is_monochromatic() && qnode_and_rnode_are_equal) ?
                        rnode->count() - 1 : rnode->count();
      densities_l_ = rnode_count *
                     global.kernel().EvalUnnormOnSq(squared_distance_range.hi);
      densities_u_ = rnode_count *
                     global.kernel().EvalUnnormOnSq(squared_distance_range.lo);
      pruned_ = rnode->count();
      used_error_ = 0.5 * (densities_u_ - densities_l_);
    }
};
}
}

#endif
