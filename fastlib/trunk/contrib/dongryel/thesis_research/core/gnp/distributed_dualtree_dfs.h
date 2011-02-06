/** @file distributed_dualtree_dfs.h
 *
 *  The prototype header for performing a distributed pairwise GNPs.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DISTRIBUTED_DUALTREE_DFS_H
#define CORE_GNP_DISTRIBUTED_DUALTREE_DFS_H

#include <boost/mpi/communicator.hpp>
#include <boost/tuple/tuple.hpp>
#include "core/gnp/dualtree_dfs.h"
#include "core/math/range.h"
#include "core/table/sub_table.h"
#include "core/table/sub_table_list.h"

namespace core {
namespace gnp {
template<typename DistributedProblemType>
class DistributedDualtreeDfs {

  public:

    /** @brief The table type.
     */
    typedef typename DistributedProblemType::TableType TableType;

    /** @brief The distributed computation problem.
     */
    typedef typename DistributedProblemType::ProblemType ProblemType;

    /** @brief The distributed table type.
     */
    typedef typename DistributedProblemType::DistributedTableType
    DistributedTableType;

    /** @brief The local tree type.
     */
    typedef typename TableType::TreeType TreeType;

    /** @brief The distributed tree type.
     */
    typedef typename DistributedTableType::TreeType DistributedTreeType;

    /** @brief The global constant type for the problem.
     */
    typedef typename DistributedProblemType::GlobalType GlobalType;

    /** @brief The type of the result being outputted.
     */
    typedef typename DistributedProblemType::ResultType ResultType;

    /** @brief The argument type of the computation.
     */
    typedef typename DistributedProblemType::ArgumentType ArgumentType;

    /** @brief The type of the object used for prioritizing the
     *         computation.
     */
    typedef typename core::gnp::DualtreeDfs<ProblemType>::FrontierObjectType
    FrontierObjectType;

    /** @brief The type of the subtable in use.
     */
    typedef core::table::SubTable<TableType> SubTableType;

    /** @brief The type of the list of subtables in use.
     */
    typedef core::table::SubTableList<SubTableType> SubTableListType;

  private:

    /** @brief The pointer to the boost communicator.
     */
    boost::mpi::communicator *world_;

    /** @brief The problem definition for the distributed computation.
     */
    DistributedProblemType *problem_;

    /** @brief The distributed query table.
     */
    DistributedTableType *query_table_;

    /** @brief The distributed reference table.
     */
    DistributedTableType *reference_table_;

    /** @brief The maximum number of points a leaf node of a local
     *         tree contains.
     */
    int leaf_size_;

    /** @brief The maximum number of tree levels to serialize at a
     *         time.
     */
    int max_num_levels_to_serialize_;

    /** @brief The maximum number of work items to dequeue per
     *         process.
     */
    int max_num_work_to_dequeue_per_stage_;

    /** @brief Some statistics about the priority queue size during
     *         the computation.
     */
    int max_computation_frontier_size_;

  private:

    /** @brief The collaborative way of exchanging items among all MPI
     *         processes for a distributed computation.
     */
    template<typename MetricType>
    void AllToAllReduce_(
      const MetricType &metric,
      typename DistributedProblemType::ResultType *query_results);

    void ResetStatisticRecursion_(
      DistributedTreeType *node, DistributedTableType * table);

    template<typename TemplateTreeType>
    void PreProcessReferenceTree_(TemplateTreeType *rnode);

    template<typename TemplateTreeType>
    void PreProcess_(TemplateTreeType *qnode);

    template<typename MetricType>
    void PostProcess_(
      const MetricType &metric,
      TreeType *qnode, ResultType *query_results);

    /** @brief The class used for prioritizing a computation object
     *         (query, reference pair).
     */
    class PrioritizeTasks_:
      public std::binary_function <
        FrontierObjectType &, FrontierObjectType &, bool > {
      public:
        bool operator()(
          const FrontierObjectType &a, const FrontierObjectType &b) const {
          return a.get<2>() > b.get<2>();
        }
    };

  public:

    /** @brief Sets the tweak parameters for the maximum number of
     *         levels of trees to grab at a time and the maximum
     *         number of work per stage to dequeue.
     */
    void set_work_params(
      int leaf_size_in,
      int max_num_levels_to_serialize_in,
      int max_num_work_to_dequeue_per_stage_in);

    /** @brief The default constructor.
     */
    DistributedDualtreeDfs();

    /** @brief Returns the associated problem.
     */
    DistributedProblemType *problem();

    /** @brief Returns the distributed query table.
     */
    DistributedTableType *query_table();

    /** @brief Returns the distributed reference table.
     */
    DistributedTableType *reference_table();

    /** @brief Resets the statistics of the query tree.
     */
    void ResetStatistic();

    /** @brief Initializes the distributed dualtree engine.
     */
    void Init(
      boost::mpi::communicator *world, DistributedProblemType &problem_in);

    /** @brief Initiates the distributed computation.
     */
    template<typename MetricType>
    void Compute(
      const MetricType &metric,
      typename DistributedProblemType::ResultType *query_results);
};
}
}

#endif
