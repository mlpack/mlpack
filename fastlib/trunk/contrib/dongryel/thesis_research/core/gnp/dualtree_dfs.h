/** @file dualtree_dfs.h
 *
 *  A template generator for performing a depth first search dual-tree
 *  algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DUALTREE_DFS_H
#define CORE_GNP_DUALTREE_DFS_H

#include <map>
#include "core/math/range.h"
#include "core/gnp/dualtree_trace.h"

namespace core {
namespace gnp {
template<typename ProblemType>
class DualtreeDfs {

  public:

    /** @brief The table type.
     */
    typedef typename ProblemType::TableType TableType;

    /** @brief The tree type.
     */
    typedef typename TableType::TreeType TreeType;

    /** @brief Global constants type for the problem.
     */
    typedef typename ProblemType::GlobalType GlobalType;

    /** @brief The type of result computed by the engine.
     */
    typedef typename ProblemType::ResultType ResultType;

  public:

    /** @brief An iterator object for iterative dual-tree computation.
     */
    template<typename IteratorMetricType>
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
              const IteratorMetricType &metric_in,
              TableType *query_table_in, TreeType *qnode_in,
              TableType *reference_table_in,
              TreeType *rnode_in);

            IteratorArgType(
              const IteratorMetricType &metric_in,
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

        const IteratorMetricType &metric_;

        ResultType *query_results_;

        core::gnp::DualtreeTrace<IteratorArgType> trace_;

      public:
        iterator(
          const IteratorMetricType &metric_in,
          DualtreeDfs<ProblemType> &engine_in,
          ResultType *query_results_in);

        void operator++();

        ResultType &operator*();

        const ResultType &operator*() const;

        void Finalize();
    };

  private:

    /** @brief The number of deterministic prunes.
     */
    int num_deterministic_prunes_;

    /** @brief The pointer to the problem.
     */
    ProblemType *problem_;

    /** @brief The query table.
     */
    TableType *query_table_;

    /** @brief Starting query node.
     */
    TreeType *query_start_node_;

    /** @brief The reference table.
     */
    TableType *reference_table_;

    bool do_selective_base_case_;

    std::map<int, int> serialize_points_per_terminal_node_;

    std::vector< std::pair<TreeType *, std::pair<int, int> > >
    unpruned_query_reference_pairs_;

    std::vector< std::pair<int, double> >
    unpruned_query_reference_pair_priorities_;

    std::map<int, int> unpruned_reference_nodes_;

  private:

    void ResetStatisticRecursion_(TreeType *node, TableType * table);

    void PreProcessReferenceTree_(TreeType *rnode);

    void PreProcess_(TreeType *qnode);

    /** @brief Performs the base case for a given node pair.
     */
    template<typename MetricType>
    void DualtreeBase_(
      const MetricType &metric,
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

    template<typename MetricType>
    bool CanProbabilisticSummarize_(
      const MetricType &metric,
      TreeType *qnode,
      TreeType *rnode,
      double failure_probability,
      typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    /** @brief Employ a probabilistic summarization with the given
     *         probability level.
     */
    void ProbabilisticSummarize_(
      GlobalType &global,
      TreeType *qnode,
      double failure_probability,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    /** @brief The heuristic for choosing one node over the other.
     */
    template<typename MetricType>
    void Heuristic_(
      const MetricType &metric,
      TreeType *node,
      TreeType *first_candidate,
      TreeType *second_candidate,
      TreeType **first_partner,
      core::math::Range &first_squared_distance_range,
      TreeType **second_partner,
      core::math::Range &second_squared_distance_range);

    /** @brief The canonical recursive case for dualtree depth-first
     *         algorithm.
     */
    template<typename MetricType>
    bool DualtreeCanonical_(
      const MetricType &metric,
      TreeType *qnode,
      TreeType *rnode,
      double failure_probability,
      const core::math::Range &squared_distance_range,
      ResultType *query_results);

    /** @brief Postprocess unaccounted contributions.
     */
    template<typename MetricType>
    void PostProcess_(
      const MetricType &metric,
      TreeType *qnode, ResultType *query_results);

  public:

    /** @brief The constructor.
     */
    DualtreeDfs();

    /** @brief Returns the list of unpruned query/reference pairs.
     */
    const std::vector <
    std::pair<TreeType *, std::pair<int, int > > > &
    unpruned_query_reference_pairs() const;

    /** @brief Returns the list of unpruned query/reference priorites.
     */
    const std::vector < std::pair<int, double > > &
    unpruned_query_reference_pair_priorities() const;

    /** @brief Returns the list of reference nodes by its beginning
     *         DFS index and the count.
     */
    const std::map< int, int > &unpruned_reference_nodes() const;

    /** @brief Sets the starting query node for the dual-tree
     *         computation.
     */
    void set_query_start_node(TreeType *query_start_node_in);

    /** @brief Sets the flag for each reference leaf node whether to
     *         compute the base case or not.
     */
    template<typename PointSerializeFlagType>
    void set_base_case_flags(
      const std::vector<PointSerializeFlagType> &flags_in);

    /** @brief Returns the number of deterministic prunes so far.
     */
    int num_deterministic_prunes() const;

    /** @brief Returns the pointer to the problem spec.
     */
    ProblemType *problem();

    /** @brief Returns the query table.
     */
    TableType *query_table();

    /** @brief Returns the reference table.
     */
    TableType *reference_table();

    /** @brief Gets an iterator object of the dualtree computation.
     */
    template<typename MetricType>
    typename DualtreeDfs<ProblemType>::template
    iterator<MetricType> get_iterator(
      const MetricType &metric_in,
      ResultType *query_results_in);

    /** @brief Resets the statistics.
     */
    void ResetStatistic();

    /** @brief Initializes the dual-tree engine with a problem spec.
     */
    void Init(ProblemType &problem_in);

    /** @brief Computes the result.
     */
    template<typename MetricType>
    void Compute(
      const MetricType &metric,
      typename ProblemType::ResultType *query_results,
      bool do_initializations = true);
};
}
}

#endif
