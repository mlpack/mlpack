/** @file dualtree_dfs.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_GNP_DUALTREE_DFS_H
#define MLPACK_GNP_DUALTREE_DFS_H

#include "core/math/range.h"
#include "dualtree_trace.h"

namespace ml {

template<typename ProblemType>
class DualtreeDfs {

  public:

    typedef typename ProblemType::TableType TableType;
    typedef typename TableType::TreeType TreeType;
    typedef typename ProblemType::ResultType ResultType;

  public:
    template<typename MetricType>
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
              const MetricType &metric_in,
              TableType *query_table_in, TreeType *qnode_in,
              TableType *reference_table_in,
              TreeType *rnode_in);

            IteratorArgType(
              const MetricType &metric_in,
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

        const MetricType &metric_;

        ResultType *query_results_;

        ml::DualtreeTrace<IteratorArgType> trace_;

      public:

        iterator(
          const MetricType &metric_in,
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

    template<typename MetricType>
    void DualtreeBase_(
      MetricType &metric,
      TreeType *qnode,
      TreeType *rnode,
      ResultType *result);

    bool CanSummarize_(
      TreeType *qnode,
      TreeType *rnode,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    template<typename MetricType>
    bool CanProbabilisticSummarize_(
      const MetricType &metric,
      TreeType *qnode,
      TreeType *rnode,
      double failure_probability,
      typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    void Summarize_(
      TreeType *qnode,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    template<typename GlobalType>
    void ProbabilisticSummarize_(
      const GlobalType &global,
      typename ProblemType::TableType::TreeType *qnode,
      double failure_probability,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    template<typename MetricType>
    void Heuristic_(
      const MetricType &metric,
      TreeType *node,
      TableType *node_table,
      TreeType *first_candidate,
      TreeType *second_candidate,
      TableType *candidate_table,
      TreeType **first_partner,
      core::math::Range &first_squared_distance_range,
      TreeType **second_partner,
      core::math::Range &second_squared_distance_range);

    template<typename MetricType>
    bool DualtreeCanonical_(
      MetricType &metric,
      TreeType *qnode,
      TreeType *rnode,
      double failure_probability,
      const core::math::Range &squared_distance_range,
      ResultType *query_results);

    template<typename MetricType>
    void PostProcess_(
      const MetricType &metric, TreeType *qnode, ResultType *query_results);

  public:

    ProblemType *problem();

    TableType *query_table();

    TableType *reference_table();

    template<typename MetricType>
    typename DualtreeDfs<ProblemType>::template
    iterator<MetricType> get_iterator(
      const MetricType &metric_in, ResultType *query_results_in);

    void ResetStatistic();

    void Init(ProblemType &problem_in);

    template<typename MetricType>
    void Compute(
      const MetricType &metric,
      typename ProblemType::ResultType *query_results);
};
};

#endif
