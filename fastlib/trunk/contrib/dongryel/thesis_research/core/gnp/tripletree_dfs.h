/** @file tripletree_dfs.h
 *
 *  A template generator for three-tree problems in depth first search
 *  mode.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_TRIPLETREE_DFS_H
#define CORE_GNP_TRIPLETREE_DFS_H

#include <boost/tuple/tuple.hpp>
#include "core/math/range.h"
#include "core/gnp/triple_range_distance_sq.h"

namespace core {
namespace gnp {

/** @brief The tripletree algorithm template generator.
 */
template<typename ProblemType>
class TripletreeDfs {

  public:

    /** @brief The table type.
     */
    typedef typename ProblemType::TableType TableType;

    /** @brief The tree type.
     */
    typedef typename TableType::TreeType TreeType;

    /** @brief The global constant type for the problem.
     */
    typedef typename ProblemType::GlobalType GlobalType;

    /** @brief The result type for the problem.
     */
    typedef typename ProblemType::ResultType ResultType;

    /** @brief The type of the object used for prioritizing the
     *         computation.
     */
    typedef boost::tuple <
    TreeType *, boost::tuple<int, int, int>, double > FrontierObjectType;

  private:

    /** @brief The pointer to the problem.
     */
    ProblemType *problem_;

    /** @brief The table object.
     */
    TableType *table_;

    /** @brief The number of deterministic prunes.
     */
    int num_deterministic_prunes_;

    /** @brief The number of Monte Carlo prunes.
     */
    int num_monte_carlo_prunes_;

  private:

    /** @brief Allocates the probability level when the recursion
     *         splits a set of nodes.
     */
    void AllocateProbabilities_(
      const std::vector<double> &failure_probabilites,
      const std::deque<bool> &node_is_split,
      const std::deque<bool> &recurse_to_left,
      const std::vector<int> &deterministic_computation_count,
      std::vector<double> *new_failure_probabilities) const;

    /** @brief Determines whether the second argument is equal to the
     *         first argument in node ID index, or comes later in DFS
     *         index.
     */
    bool NodeIsAgreeable_(TreeType *node, TreeType *next_node) const;

    typename TableType::TreeIterator GetNextNodeIterator_(
      const core::gnp::TripleRangeDistanceSq<TableType> &range_sq_in,
      int node_index,
      const typename TableType::TreeIterator &it_in);

    void ResetStatisticRecursion_(TreeType *node);

    /** @brief A helper function for generating all triple-tree
     *         combination for a set of three nodes.
     */
    template<typename MetricType>
    void RecursionHelper_(
      const MetricType &metric,
      core::gnp::TripleRangeDistanceSq<TableType> &triple_range_distance_sq,
      double relative_error,
      const std::vector<double> &failure_probabilities,
      typename ProblemType::ResultType *query_results,
      int level,
      bool all_leaves,
      std::deque<bool> &node_is_split,
      std::deque<bool> &recurse_to_left,
      std::vector<int> &deterministic_computation_count,
      bool *deterministic_approximation);

    void PreProcess_(TreeType *node);

    /** @brief Computes a triple of nodes naively.
     */
    template<typename MetricType>
    void TripletreeBase_(
      const MetricType &metric,
      const core::gnp::TripleRangeDistanceSq<TableType> &range_in,
      ResultType *result);

    template<typename MetricType>
    bool CanProbabilisticSummarize_(
      const MetricType &metric,
      const core::gnp::TripleRangeDistanceSq<TableType> &range_in,
      const std::vector<double> &failure_probabilities,
      int node_start_index,
      typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    template<typename MetricType>
    void ProbabilisticSummarize_(
      const MetricType &metric,
      GlobalType &global,
      const core::gnp::TripleRangeDistanceSq<TableType> &range_in,
      const std::vector<double> &failure_probabilities,
      int probabilistic_node_start_index,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    bool CanSummarize_(
      const core::gnp::TripleRangeDistanceSq<TableType> &range_in,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results,
      int *failure_index);

    void Summarize_(
      const core::gnp::TripleRangeDistanceSq<TableType> &range_in,
      int probabilistic_start_node_index,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    /** @brief The canonical recursion function for tripletree
     *         algorithms.
     */
    template<typename MetricType>
    bool TripletreeCanonical_(
      const MetricType &metric,
      core::gnp::TripleRangeDistanceSq<TableType> &triple_range_distance_sq,
      double relative_error,
      const std::vector<double> &failure_probabilities,
      typename ProblemType::ResultType *query_results);

    template<typename MetricType>
    void PostProcess_(
      const MetricType &metric,
      TreeType *node, ResultType *query_results,
      bool do_query_results_postprocess);

  public:

    /** @brief Returns the number of deterministic prunes.
     */
    int num_deterministic_prunes() const;

    /** @brief Returns the number of Monte Carlo prunes.
     */
    int num_monte_carlo_prunes() const;

    /** @brief Returns the pointer to the problem.
     */
    ProblemType *problem();

    /** @brief Returns the pointer to the table.
     */
    TableType *table();

    void ResetStatistic();

    /** @brief Initializes the triple tree engine with a problem
     *         specification.
     */
    void Init(ProblemType &problem_in);

    /** @brief Computes the triple tree problem naively.
     */
    template<typename MetricType>
    void NaiveCompute(
      const MetricType &metric,
      typename ProblemType::ResultType *naive_query_results);

    /** @brief Computes the triple tree problem using a tree-based
     *         approach.
     */
    template<typename MetricType>
    void Compute(
      const MetricType &metric,
      typename ProblemType::ResultType *query_results);
};
}
}

#endif
