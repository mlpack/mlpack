/** @file distributed_dualtree_dfs.h
 *
 *  The prototype header for performing a distributed pairwise GNPs.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_DFS_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_DFS_H

#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/tuple/tuple.hpp>
#include "core/gnp/dualtree_dfs.h"
#include "core/math/range.h"
#include "core/parallel/distributed_dualtree_task.h"
#include "core/parallel/distributed_dualtree_task_queue.h"
#include "core/parallel/route_request.h"
#include "core/parallel/table_exchange.h"
#include "core/table/sub_table.h"
#include <omp.h>

namespace core {
namespace parallel {

/** @brief The type of the priority queue that is used for
 *         prioritizing the fine-grained computations.
 */
template<typename TableType>
class FinePriorityQueue {
  public:

    /** @brief The type of the object used for prioritizing the
     *         computation per stage on a shared memory (on a fine
     *         scale).
     */
    typedef core::parallel::DistributedDualtreeTask <
    TableType > FineFrontierObjectType;

    /** @brief The reference count used for the intrusive pointer.
     */
    long reference_count_;

  private:

    /** @brief The class used for prioritizing a computation object
     *         (query, reference pair).
     */
    template<typename FrontierObjectType>
    class PrioritizeTasks_:
      public std::binary_function <
        FrontierObjectType &, FrontierObjectType &, bool > {
      public:
        bool operator()(
          const FrontierObjectType &a, const FrontierObjectType &b) const {
          return a.priority() < b.priority();
        }
    };

    std::priority_queue <
    FineFrontierObjectType,
    std::vector<FineFrontierObjectType>,
    PrioritizeTasks_<FineFrontierObjectType> > queue_;

  public:

    typedef typename std::priority_queue <
    FineFrontierObjectType,
    std::vector<FineFrontierObjectType>,
    PrioritizeTasks_<FineFrontierObjectType> >::value_type value_type;

  public:

    void pop() {
      queue_.pop();
    }

    int size() const {
      return queue_.size();
    }

    FinePriorityQueue() {
      reference_count_ = 0;
    }

    const value_type &top() const {
      return queue_.top();
    }

    void push(const value_type &value_in) {
      queue_.push(value_in);
    }
};

template<typename TableType>
inline void intrusive_ptr_add_ref(FinePriorityQueue<TableType> *ptr) {
  ptr->reference_count_++;
}

template<typename TableType>
inline void intrusive_ptr_release(FinePriorityQueue<TableType> *ptr) {
  ptr->reference_count_--;
  if(ptr->reference_count_ == 0) {
    if(core::table::global_m_file_) {
      core::table::global_m_file_->DestroyPtr(ptr);
    }
    else {
      delete ptr;
    }
  }
}

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
     *         computation per stage on a shared memory (on a fine
     *         scale).
     */
    typedef core::parallel::DistributedDualtreeTask <
    TableType > FineFrontierObjectType;

    /** @brief The type of the subtable in use.
     */
    typedef core::table::SubTable<TableType> SubTableType;

    /** @brief The type of the ID of subtables.
     */
    typedef typename SubTableType::SubTableIDType SubTableIDType;

    /** @brief The type of the priority queue used for prioritizing
     *         fine-grained computations.
     */
    typedef class FinePriorityQueue<TableType> FinePriorityQueueType;

  private:

    /** @brief Whether to perform load balancing.
     */
    bool do_load_balancing_;

    /** @brief The maximum number of points a leaf node of a local
     *         tree contains.
     */
    int leaf_size_;

    /** @brief The maximum size of the subtree to serialize at a time.
     */
    int max_subtree_size_;

    /** @brief The maximum number of work items to dequeue per
     *         process.
     */
    int max_num_work_to_dequeue_per_stage_;

    /** @brief This is used for weak-scaling, limiting the number of
     *         reference points considered for each query point.
     */
    unsigned long int max_num_reference_points_to_pack_per_process_;

    /** @brief The number of deterministic prunes.
     */
    int num_deterministic_prunes_;

    /** @brief The number of probabilistic prunes.
     */
    int num_probabilistic_prunes_;

    /** @brief The problem definition for the distributed computation.
     */
    DistributedProblemType *problem_;

    /** @brief The distributed query table.
     */
    DistributedTableType *query_table_;

    /** @brief The distributed reference table.
     */
    DistributedTableType *reference_table_;

    /** @brief Used for determining the maximum number of reference
     *         points to pack per process, if weak-scaling measuring
     *         mode is enabled.
     */
    double weak_scaling_factor_;

    /** @brief Whether the distributed computation is for measuring
     *         weak-scalability.
     */
    bool weak_scaling_measuring_mode_;

    /** @brief The pointer to the boost communicator.
     */
    boost::mpi::communicator *world_;

  private:

    typedef core::parallel::DistributedDualtreeTaskQueue <
    DistributedTableType,
    FinePriorityQueueType, ProblemType > DistributedDualtreeTaskQueueType;

    template<typename MetricType>
    void ComputeEssentialReferenceSubtrees_(
      const MetricType &metric_in,
      int max_reference_subtree_size,
      DistributedTreeType *global_query_node,
      TreeType *local_reference_node,
      std::vector< std::vector< std::pair<int, int> > > *
      essential_reference_subtrees,
      std::vector <
      core::parallel::RouteRequest<SubTableType> > *
      hashed_essential_reference_subtrees,
      std::vector< unsigned long int > *
      num_reference_points_assigned_per_process,
      std::vector< std::vector< core::math::Range> > *
      remote_priorities,
      std::vector<unsigned long int> *extrinsic_prunes);

    template<typename MetricType>
    void InitialSetup_(
      const MetricType &metric,
      typename DistributedProblemType::ResultType *query_results,
      std::vector <
      core::parallel::RouteRequest<SubTableType> >
      *hashed_essential_reference_subtress_to_send,
      DistributedDualtreeTaskQueueType *distributed_tasks);

    /** @brief The collaborative way of exchanging items among all MPI
     *         processes for a distributed computation. This routine
     *         utilizes asynchronous MPI calls to maximize
     *         communication and computation overlap.
     */
    template<typename MetricType>
    void AllToAllIReduce_(
      const MetricType &metric,
      boost::mpi::timer *timer,
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

  public:

    /** @brief Returns the number of deterministic prunes so far.
     */
    int num_deterministic_prunes() const;

    /** @brief Returns the number of probabilistic prunes so far.
     */
    int num_probabilistic_prunes() const;

    /** @brief Sets the tweak parameters for the maximum number of
     *         levels of trees to grab at a time and the maximum
     *         number of work per stage to dequeue.
     */
    void set_work_params(
      int leaf_size_in,
      int max_subtree_size_in,
      bool do_load_balancing_in,
      int max_num_work_to_dequeue_per_stage_in);

    /** @brief Enables weak scaling measuring mode.
     */
    void enable_weak_scaling_measuring_mode(double factor);

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
