/** @file tripletree_dfs.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_TRIPLETREE_DFS_H
#define CORE_GNP_TRIPLETREE_DFS_H

#include "core/metric_kernels/abstract_metric.h"
#include "core/math/range.h"
#include "core/gnp/triple_range_distance_sq.h"

namespace core {
namespace gnp {
template<typename ProblemType>
class TripletreeDfs {

  public:

    typedef typename ProblemType::TableType TableType;
    typedef typename TableType::TreeType TreeType;
    typedef typename ProblemType::ResultType ResultType;

  private:

    ProblemType *problem_;

    TableType *table_;

  private:

    typename TableType::TreeIterator GetNextNodeIterator_(
      const core::gnp::TripleRangeDistanceSq &range_sq_in,
      int node_index,
      const typename TableType::TreeIterator &it_in);

    void ResetStatisticRecursion_(TreeType *node);

    void PreProcess_(TreeType *node);

    void TripletreeBase_(
      const core::metric_kernels::AbstractMetric &metric,
      const core::gnp::TripleRangeDistanceSq &range_in,
      ResultType *result);

    bool CanSummarize_(
      const core::gnp::TripleRangeDistanceSq &range_in,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    void Summarize_(
      const core::gnp::TripleRangeDistanceSq &range_in,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    bool TripletreeCanonical_(
      const core::metric_kernels::AbstractMetric &metric,
      const core::gnp::TripleRangeDistanceSq &triple_range_distance_sq,
      double relative_error,
      double probability,
      typename ProblemType::ResultType *query_results);

    void PostProcess_(
      const core::metric_kernels::AbstractMetric &metric,
      TreeType *node, ResultType *query_results);

  public:

    ProblemType *problem();

    TableType *table();

    void ResetStatistic();

    void Init(ProblemType &problem_in);

    void Compute(
      const core::metric_kernels::AbstractMetric &metric,
      typename ProblemType::ResultType *query_results);
};
};
};

#endif
