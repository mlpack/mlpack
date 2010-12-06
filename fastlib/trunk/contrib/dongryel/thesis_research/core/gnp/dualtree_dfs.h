/** @file dualtree_dfs.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DUALTREE_DFS_H
#define CORE_GNP_DUALTREE_DFS_H

#include "core/metric_kernels/abstract_metric.h"
#include "core/math/range.h"
#include "dualtree_trace.h"

namespace core {
namespace gnp {
template<typename ProblemType>
class DualtreeDfs {

  public:

    typedef typename ProblemType::TableType TableType;
    typedef typename TableType::TreeType TreeType;
    typedef typename ProblemType::GlobalType GlobalType;
    typedef typename ProblemType::ResultType ResultType;

  public:
    class iterator {
      private:
        class IteratorArgType {

          private:

            TreeType *qnode_;

            TreeType *rnode_;

            core::math::Range squared_distance_range_;

          public:

            IteratorArgType();

            IteratorArgType(const IteratorArgType &arg_in);

            IteratorArgType(
              const core::metric_kernels::AbstractMetric &metric_in,
              TableType *query_table_in, TreeType *qnode_in,
              TableType *reference_table_in,
              TreeType *rnode_in);

            IteratorArgType(
              const core::metric_kernels::AbstractMetric &metric_in,
              TableType *query_table_in, TreeType *qnode_in,
              TableType *reference_table_in,
              TreeType *rnode_in,
              const core::math::Range &squared_distance_range_in);

            TreeType *qnode();

            TreeType *qnode() const;

            TreeType *rnode();

            TreeType *rnode() const;

            const core::math::Range &squared_distance_range() const;
        };

      private:

        TableType *query_table_;

        TableType *reference_table_;

        DualtreeDfs<ProblemType> *engine_;

        const core::metric_kernels::AbstractMetric &metric_;

        ResultType *query_results_;

        ml::DualtreeTrace<IteratorArgType> trace_;

      public:

        iterator(
          const core::metric_kernels::AbstractMetric &metric_in,
          DualtreeDfs<ProblemType> &engine_in,
          ResultType *query_results_in);

        void operator++();

        ResultType &operator*();

        const ResultType &operator*() const;

        void Finalize();
    };

  private:

    ProblemType *problem_;

    TableType *query_table_;

    TableType *reference_table_;

  private:

    void ResetStatisticRecursion_(TreeType *node, TableType * table);

    void PreProcessReferenceTree_(TreeType *rnode);

    void PreProcess_(TreeType *qnode);

    void DualtreeBase_(
      const core::metric_kernels::AbstractMetric &metric,
      TreeType *qnode,
      TreeType *rnode,
      ResultType *result);

    bool CanSummarize_(
      TreeType *qnode,
      TreeType *rnode,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    void Summarize_(
      TreeType *qnode,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    bool CanProbabilisticSummarize_(
      const core::metric_kernels::AbstractMetric &metric,
      TreeType *qnode,
      TreeType *rnode,
      double failure_probability,
      typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    void ProbabilisticSummarize_(
      GlobalType &global,
      TreeType *qnode,
      double failure_probability,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    void Heuristic_(
      const core::metric_kernels::AbstractMetric &metric,
      TreeType *node,
      TableType *node_table,
      TreeType *first_candidate,
      TreeType *second_candidate,
      TableType *candidate_table,
      TreeType **first_partner,
      core::math::Range &first_squared_distance_range,
      TreeType **second_partner,
      core::math::Range &second_squared_distance_range);

    bool DualtreeCanonical_(
      const core::metric_kernels::AbstractMetric &metric,
      TreeType *qnode,
      TreeType *rnode,
      double failure_probability,
      const core::math::Range &squared_distance_range,
      ResultType *query_results);

    void PostProcess_(
      const core::metric_kernels::AbstractMetric &metric,
      TreeType *qnode, ResultType *query_results);

  public:

    ProblemType *problem();

    TableType *query_table();

    TableType *reference_table();

    typename DualtreeDfs<ProblemType>::iterator get_iterator(
      const core::metric_kernels::AbstractMetric &metric_in,
      ResultType *query_results_in);

    void ResetStatistic();

    void Init(ProblemType &problem_in);

    void Compute(
      const core::metric_kernels::AbstractMetric &metric,
      typename ProblemType::ResultType *query_results,
      bool do_initializations = true);
};
};
};

#endif
