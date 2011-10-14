/** @file distributed_dualtree_dfs_dev.h
 *
 *  The generic algorithm template for distributed dual-tree
 *  computation.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_DFS_DEV_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_DFS_DEV_H

#include <boost/bind.hpp>
#include <boost/mpi.hpp>
#include <boost/tuple/tuple.hpp>
#include <map>
#include <queue>
#include "core/parallel/distributed_dualtree_dfs.h"
#include "core/parallel/distributed_dualtree_task_queue.h"
#include "core/gnp/dualtree_dfs_dev.h"
#include "core/parallel/message_tag.h"
#include "core/table/table.h"
#include "core/table/memory_mapped_file.h"

namespace core {
namespace parallel {

template<typename ProblemType>
int DistributedDualtreeDfs<ProblemType>::num_deterministic_prunes() const {
  return num_deterministic_prunes_;
}

template<typename ProblemType>
int DistributedDualtreeDfs<ProblemType>::num_probabilistic_prunes() const {
  return num_probabilistic_prunes_;
}

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs<DistributedProblemType>::AllToAllIReduce_(
  const MetricType &metric,
  double *tree_walk_time,
  typename DistributedProblemType::ResultType *query_results) {

  // Figure out the list of reference subtrees to send and to receive.
  std::vector< std::vector< std::pair<int, int> > >
  reference_subtrees_to_send(world_->size());
  std::vector <
  std::vector< core::math::Range> > send_priorities(world_->size());
  std::vector <
  std::vector< std::pair<int, int> > > reference_subtrees_to_receive;
  std::vector< std::vector< core::math::Range> > receive_priorities;

  // The list of prioritized tasks this MPI process needs to take care
  // of.
  DistributedDualtreeTaskQueueType distributed_tasks;

  // The number of reference subtrees to receive and to send in total.
  std::vector <
  core::parallel::RouteRequest<SubTableType> >
  hashed_essential_reference_subtrees_to_send;

  // For each process, initialize the distributed task object.
  distributed_tasks.Init(
    metric, *world_, max_subtree_size_, do_load_balancing_,
    query_table_, reference_table_, query_results,
    omp_get_max_threads(), weak_scaling_measuring_mode_,
    max_num_reference_points_to_pack_per_process_);

  // Walk the tree before entering parallel.
  distributed_tasks.WalkReferenceTree(
    metric, problem_->global(), *world_,
    &hashed_essential_reference_subtrees_to_send);

  // OpenMP parallel region. The master thread is the only one that is
  // allowed to make MPI calls (sending and receiving reference
  // subtables).
  #pragma omp parallel
  {

    // The thread ID.
    int thread_id = omp_get_thread_num();

    // The number of tasks to dequeue for this thread. The policy is
    // that the master thread dequeues somewhat a smaller number of
    // tasks, while the slaves do a deeper lock, since the master
    // thread makes all MPI calls.
    int num_tasks_to_dequeue =
      (omp_get_num_threads() > 1 && thread_id == 0 && world_->size() > 1) ?
      1 : max_num_work_to_dequeue_per_stage_;

    // Used for determining the termination condition.
    bool work_left_to_do = true;

    do {

      // The task found in the current iteration.
      std::pair< std::vector<FineFrontierObjectType> , int> found_task;
      found_task.first.resize(0);
      found_task.second = -1;

      // Only the master thread makes MPI calls.
      if(thread_id == 0 && world_->size() > 1) {
        distributed_tasks.SendReceive(
          metric, *world_, hashed_essential_reference_subtrees_to_send);
      }

      // After enqueing, everyone else tries to dequeue the tasks.
      typename DistributedDualtreeTaskQueueType::
      QuerySubTableLockListType::iterator checked_out_query_subtable;
      distributed_tasks.DequeueTask(
        *world_, thread_id, metric,
        & hashed_essential_reference_subtrees_to_send,
        problem_->global(), num_tasks_to_dequeue,
        &found_task, &checked_out_query_subtable);

      // If found something to run on, then call the serial dual-tree
      // method.
      if(found_task.second >= 0) {

        for(unsigned int i = 0; i < found_task.first.size(); i++) {

          // Run a sub-dualtree algorithm on the computation object.
          core::gnp::DualtreeDfs<ProblemType> sub_engine;
          ProblemType sub_problem;
          ArgumentType sub_argument;
          TableType *task_query_table =
            found_task.first[i].query_subtable().table();
          TableType *task_reference_table =
            found_task.first[i].reference_subtable().table();
          TreeType *task_starting_rnode =
            found_task.first[i].reference_start_node();
          int task_reference_cache_id =
            found_task.first[i].reference_subtable_cache_block_id();

          // Initialize the argument, the problem, and the engine.
          sub_argument.Init(
            task_reference_table, task_query_table, problem_->global());
          sub_problem.Init(sub_argument, &(problem_->global()));
          sub_engine.Init(sub_problem);

          // Set the starting query node.
          sub_engine.set_query_start_node(
            found_task.first[i].query_start_node());

          // Set the starting reference node.
          sub_engine.set_reference_start_node(task_starting_rnode);

          // Fire away the computation.
          //typename core::gnp::DualtreeDfs<ProblemType>::template iterator< MetricType > sub_engine_it =
          //sub_engine.get_iterator(
          //  metric,
          //  found_task.first[i].query_result());
          //while(++sub_engine_it) {
          //}

          sub_engine.Compute(
            metric, found_task.first[i].query_result(), false);

          // Synchronize the sub-result with the MPI result owned by the
          // current process.
          distributed_tasks.PostComputeSynchronize(
            sub_engine.num_deterministic_prunes(),
            sub_engine.num_probabilistic_prunes(),
            *(found_task.first[i].query_result()), query_results);

          // Push in the completed amount of work.
          unsigned long int completed_work =
            static_cast<unsigned long int>(
              found_task.first[i].query_start_node()->count()) *
            static_cast<unsigned long int>(task_starting_rnode->count());
          distributed_tasks.push_completed_computation(
            * world_, task_starting_rnode->count(),
            completed_work, checked_out_query_subtable);

          // Release the reference subtable.
          distributed_tasks.ReleaseCache(* world_, task_reference_cache_id, 1);
        } // end of taking care of each task.

        // After finishing, the lock on the query subtree is released.
        distributed_tasks.ReturnQuerySubTable(
          * world_, checked_out_query_subtable);

      } // end of finding a task.

      // Quit if all work is done.
      work_left_to_do = !(distributed_tasks.can_terminate(* world_));
    }
    while(work_left_to_do);

  } // end of omp parallel

  // Extract the prune counts.
  num_deterministic_prunes_ = distributed_tasks.num_deterministic_prunes();
  num_probabilistic_prunes_ = distributed_tasks.num_probabilistic_prunes();

  // Extract the tree walk time.
  * tree_walk_time = distributed_tasks.tree_walk_time();

  // Do a global post-processing if necessary over the set of global
  // query results owned by each MPI process.
  query_results->PostProcess(* world_, query_table_);
}

template<typename DistributedProblemType>
DistributedProblemType *DistributedDualtreeDfs <
DistributedProblemType >::problem() {
  return problem_;
}

template<typename DistributedProblemType>
typename DistributedProblemType::DistributedTableType *
DistributedDualtreeDfs<DistributedProblemType>::query_table() {
  return query_table_;
}

template<typename DistributedProblemType>
typename DistributedProblemType::DistributedTableType *
DistributedDualtreeDfs<DistributedProblemType>::reference_table() {
  return reference_table_;
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs <
DistributedProblemType >::ResetStatistic() {
  ResetStatisticRecursion_(query_table_->get_tree(), query_table_);
}

template<typename DistributedProblemType>
DistributedDualtreeDfs<DistributedProblemType>::DistributedDualtreeDfs() {
  do_load_balancing_ = false;
  leaf_size_ = 0;
  max_num_work_to_dequeue_per_stage_ = 5;
  max_num_reference_points_to_pack_per_process_ =
    std::numeric_limits<unsigned long int>::max();
  max_subtree_size_ = 20000;
  num_deterministic_prunes_ = 0;
  num_probabilistic_prunes_ = 0;
  weak_scaling_factor_ = 0.0;
  weak_scaling_measuring_mode_ = false;
  world_ = NULL;
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs <
DistributedProblemType >::enable_weak_scaling_measuring_mode(double factor_in) {
  weak_scaling_factor_ = factor_in;
  weak_scaling_measuring_mode_ = true;
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs<DistributedProblemType>::set_work_params(
  int leaf_size_in,
  int max_subtree_size_in,
  bool do_load_balancing_in,
  int max_num_work_to_dequeue_per_stage_in) {

  leaf_size_ = leaf_size_in;
  max_subtree_size_ = max_subtree_size_in;
  do_load_balancing_ = do_load_balancing_in;
  max_num_work_to_dequeue_per_stage_ = max_num_work_to_dequeue_per_stage_in;
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs<DistributedProblemType>::Init(
  boost::mpi::communicator *world_in,
  DistributedProblemType &problem_in) {
  world_ = world_in;
  problem_ = &problem_in;
  query_table_ = problem_->query_table();
  reference_table_ = problem_->reference_table();
  ResetStatistic();

  if(query_table_ != reference_table_) {
    ResetStatisticRecursion_(reference_table_->get_tree(), reference_table_);
  }
}

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs<DistributedProblemType>::Compute(
  const MetricType &metric,
  typename DistributedProblemType::ResultType *query_results) {

  // If weak-scaling measure mode is enabled, re-adjust the max_subtree_size_in.
  if(weak_scaling_measuring_mode_) {

    // Compute the average number of query points across all MPI processes.
    unsigned long int avg_num_query_points_per_process = 0;
    for(int i = 0; i < world_->size(); i++) {
      avg_num_query_points_per_process +=
        problem_->query_table()->local_n_entries(i) ;
    }
    avg_num_query_points_per_process /= (world_->size());

    // Given the factor, compute the maximum number of reference
    // points to pack per each reference MPI process.
    max_num_reference_points_to_pack_per_process_ =
      (world_->size() == 1) ?
      static_cast<unsigned long int>(
        weak_scaling_factor_ *
        (avg_num_query_points_per_process /  omp_get_max_threads())) :
      static_cast<unsigned long int>(
        weak_scaling_factor_ *
        (avg_num_query_points_per_process /
         (2 * log(world_->size()) * omp_get_max_threads())));

    // Re-adjust so that the subtree is transfered in around 3 rounds.
    max_subtree_size_ =
      std::max(
        std::min(
          static_cast<int>(max_num_reference_points_to_pack_per_process_) / 3,
          20000),
        leaf_size_ * 2);

    if(world_->rank() == 0) {
      std::cerr << "Measuring weak-scalability...\n";
      std::cerr << "Each query MPI process will need to consider " <<
                weak_scaling_factor_ * 100 << " \% of the average query points " <<
                "as the number of reference points per each query point...\n";
      std::cerr << "Each query MPI process will encounter " <<
                max_num_reference_points_to_pack_per_process_ <<
                " reference points per reference process.\n";
      std::cerr << "Readjusting the --max_subtree_size_in to " <<
                max_subtree_size_ << "...\n";
    }
  }

  if(world_->rank() == 0) {
    if(do_load_balancing_) {
      std::cerr << "Dynamic load-balancing option is turned ON.\n";
    }
    else {
      std::cerr << "Dynamic load-balancing option is turned OFF.\n";
    }
  }

  // The MPI timer.
  boost::mpi::timer timer;

  // Allocate space for storing the final results.
  query_results->Init(problem_->global(), query_table_->n_entries());

  // Preprocess the global query tree and the local query tree owned
  // by each process.
  PreProcess_(query_table_->get_tree());
  PreProcess_(query_table_->local_table()->get_tree());

  // Preprocess the global reference tree, and the local reference
  // tree owned by each process. This part needs to be fixed so that
  // it does a true bottom-up refinement using an MPI-gather.
  core::gnp::DualtreeDfs<ProblemType> self_engine;
  ProblemType self_problem;
  ArgumentType self_argument;
  self_argument.Init(problem_->global());
  self_problem.Init(self_argument, &(problem_->global()));
  self_engine.Init(self_problem);
  core::gnp::DualtreeDfs<ProblemType>::PreProcessReferenceTree(
    self_engine.problem()->global(),
    reference_table_->local_table()->get_tree());
  PreProcessReferenceTree_(reference_table_->get_tree());

  // Figure out each process's work using the global tree. Currently
  // only supports P = power of two. Fix this later.
  if(world_->size() & (world_->size() - 1)) {
    if(world_->rank() == 0) {
      std::cerr << "Re-run with the number of processes equal to a power of "
                << "two!\n";
    }
    return;
  }

  // Call the tree-based reduction.
  double tree_walk_time;
  AllToAllIReduce_(metric, &tree_walk_time, query_results);
  double elapsed_time = timer.elapsed();
  std::vector<double> collected_elapsed_times;
  boost::mpi::gather(*world_, elapsed_time, collected_elapsed_times, 0);
  std::vector<double> collected_tree_walk_times;
  boost::mpi::gather(*world_, tree_walk_time, collected_tree_walk_times, 0);
  std::pair<int, int> num_prunes(
    this->num_deterministic_prunes(), this->num_probabilistic_prunes());
  std::vector< std::pair<int, int> > collected_num_prunes;
  boost::mpi::gather(*world_, num_prunes, collected_num_prunes, 0);
  if(world_->rank() == 0) {
    for(int i = 0; i < world_->size(); i++) {
      std::cerr << "Process " << i << " took " <<
                collected_tree_walk_times[i] << " seconds to walk the tree and " <<
                collected_elapsed_times[i] << " seconds to compute.\n";
      std::cerr << "  Deterministic prunes: " <<
                collected_num_prunes[i].first << " , probabilistic prunes: " <<
                collected_num_prunes[i].second << "\n";
    }
  }
  world_->barrier();
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs <
DistributedProblemType >::ResetStatisticRecursion_(
  typename DistributedProblemType::DistributedTableType::TreeType *node,
  typename DistributedProblemType::DistributedTableType * table) {
  node->stat().SetZero();
  if(node->is_leaf() == false) {
    ResetStatisticRecursion_(node->left(), table);
    ResetStatisticRecursion_(node->right(), table);
  }
}

template<typename DistributedProblemType>
template<typename TemplateTreeType>
void DistributedDualtreeDfs <
DistributedProblemType >::PreProcessReferenceTree_(
  TemplateTreeType *rnode) {

  // Does not do anything yet.
}

template<typename DistributedProblemType>
template<typename TemplateTreeType>
void DistributedDualtreeDfs<DistributedProblemType>::PreProcess_(
  TemplateTreeType *qnode) {

  typename DistributedProblemType::StatisticType &qnode_stat = qnode->stat();
  qnode_stat.SetZero();

  if(! qnode->is_leaf()) {
    PreProcess_(qnode->left());
    PreProcess_(qnode->right());
  }
}
}
}

#endif
