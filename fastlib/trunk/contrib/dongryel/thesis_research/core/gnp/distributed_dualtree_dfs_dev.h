/** @file distributed_dualtree_dfs_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DISTRIBUTED_DUALTREE_DFS_DEV_H
#define CORE_GNP_DISTRIBUTED_DUALTREE_DFS_DEV_H

#include <boost/mpi.hpp>
#include "core/gnp/distributed_dualtree_dfs.h"
#include "core/gnp/dualtree_dfs_dev.h"
#include "core/table/table.h"
#include "core/table/memory_mapped_file.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
};
};

template<typename DistributedProblemType>
void core::gnp::DistributedDualtreeDfs<DistributedProblemType>::AllReduce_(
  const core::metric_kernels::AbstractMetric &metric,
  typename DistributedProblemType::ResultType *query_results) {

  // Start the computation with the self interaction.
  core::gnp::DualtreeDfs<ProblemType> self_engine;
  ProblemType self_problem;
  ArgumentType self_argument;
  self_argument.Init(problem_->global());
  self_problem.Init(self_argument);
  self_engine.Init(self_problem);
  self_engine.Compute(metric, query_results);
  world_->barrier();

  // Set the local table.
  std::vector< TableType * > remote_tables(world_->size(), NULL);
  remote_tables[ world_->rank()] = reference_table_->local_table();

  // Pair up processes, and exchange. Right now, the entire local
  // reference tree is exchanged between pairs of processes, but this
  // could be improved later. Also assume that the number of processes
  // is a power of two for the moment. This will be changed later to allow
  // transfer of data at a finer granularity, which means there will
  // be an outer loop over the main all-reduce. This solution also is
  // not topology-aware, so it will be changed later to fit the
  // appropriate network topology.
  int num_rounds = log2(world_->size());
  for(int r = 1; r <= num_rounds; r++) {
    int stride = 1 << r;
    int num_tables_in_action = stride >> 1;

    // Exchange with the appropriate process.
    int group_offset = world_->rank() % stride;
    int group_leader = world_->rank() - group_offset;
    int group_end = group_leader + stride - 1;
    int exchange_process_id = group_leader + stride - group_offset - 1;

    // Send the process's own collected tables.
    std::vector<boost::mpi::request> send_requests;
    std::vector<int> received_tables_in_current_iter;
    send_requests.resize(num_tables_in_action);
    for(int i = 0; i < num_tables_in_action; i++) {
      int send_id;
      if(world_->rank() - group_leader < group_end - world_->rank()) {
        send_id = group_leader + i;
      }
      else {
        send_id = group_leader + i + num_tables_in_action;
      }
      send_requests[i] = world_->isend(
                           exchange_process_id, send_id,
                           *(remote_tables[send_id]));
    }
    for(int i = 0; i < num_tables_in_action; i++) {
      int receive_id;
      if(world_->rank() - group_leader < group_end - world_->rank()) {
        receive_id = group_leader + i + num_tables_in_action;
      }
      else {
        receive_id = group_leader + i;
      }
      received_tables_in_current_iter.push_back(receive_id);
      remote_tables[receive_id] =
        core::table::global_m_file_->Construct<TableType>();
      world_->recv(
        exchange_process_id, receive_id, *(remote_tables[receive_id]));
    }
    boost::mpi::wait_all(send_requests.begin(), send_requests.end());

    // Each process calls the independent sets of serial dual-tree dfs
    // algorithms. Further parallelism can be exploited here.
    for(unsigned i = 0; i < received_tables_in_current_iter.size(); i++) {
      core::gnp::DualtreeDfs<ProblemType> sub_engine;
      ProblemType sub_problem;
      ArgumentType sub_argument;
      sub_argument.Init(
        remote_tables[received_tables_in_current_iter[i]],
        query_table_->local_table(),
        problem_->global());
      sub_problem.Init(sub_argument);
      sub_engine.Init(sub_problem);
      sub_engine.Compute(metric, query_results, false);
    }
    world_->barrier();

  } // End of the all-reduce loop.

  // Destroy all tables after all computations are done, except for
  // the process's own table.
  for(int i = 0; i < static_cast<int>(remote_tables.size()); i++) {
    if(i != world_->rank()) {
      core::table::global_m_file_->DestroyPtr(remote_tables[i]);
    }
  }
}

template<typename DistributedProblemType>
DistributedProblemType *core::gnp::DistributedDualtreeDfs<DistributedProblemType>::problem() {
  return problem_;
}

template<typename DistributedProblemType>
typename DistributedProblemType::DistributedTableType *
core::gnp::DistributedDualtreeDfs<DistributedProblemType>::query_table() {
  return query_table_;
}

template<typename DistributedProblemType>
typename DistributedProblemType::DistributedTableType *
core::gnp::DistributedDualtreeDfs<DistributedProblemType>::reference_table() {
  return reference_table_;
}

template<typename DistributedProblemType>
void core::gnp::DistributedDualtreeDfs<DistributedProblemType>::ResetStatistic() {
  ResetStatisticRecursion_(query_table_->get_tree(), query_table_);
}

template<typename DistributedProblemType>
void core::gnp::DistributedDualtreeDfs<DistributedProblemType>::Init(
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
void core::gnp::DistributedDualtreeDfs<DistributedProblemType>::Compute(
  const core::metric_kernels::AbstractMetric &metric,
  typename DistributedProblemType::ResultType *query_results) {

  // Allocate space for storing the final results.
  query_results->Init(query_table_->n_entries());

  // Preprocess the global query tree and the local query tree owned
  // by each process.
  PreProcess_(query_table_->get_tree());
  PreProcess_(query_table_->local_table()->get_tree());

  // Preprocess the global reference tree, and the local reference
  // tree owned by each process. This part needs to be fixed so that
  // it does a true bottom-up refinement using an MPI-gather.
  PreProcessReferenceTree_(reference_table_->get_tree());
  PreProcessReferenceTree_(reference_table_->local_table()->get_tree());

  // Figure out each process's work using the global tree. This is
  // done using a naive approach where the global goal is to complete
  // a 2D matrix workspace. This is currently doing an all-reduce type
  // of exchange.
  AllReduce_(metric, query_results);
  world_->barrier();

  // Postprocess.
  // PostProcess_(metric, query_table_->get_tree(), query_results);
}

template<typename DistributedProblemType>
void core::gnp::DistributedDualtreeDfs<DistributedProblemType>::ResetStatisticRecursion_(
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
void core::gnp::DistributedDualtreeDfs<DistributedProblemType>::PreProcessReferenceTree_(
  TemplateTreeType *rnode) {

  typename DistributedProblemType::StatisticType &rnode_stat = rnode->stat();
  typename DistributedProblemType::DistributedTableType::TreeIterator rnode_it =
    reference_table_->get_node_iterator(rnode);

  if(rnode->is_leaf()) {
    rnode_stat.Init(rnode_it);
  }
  else {

    // Get the left and the right children.
    TemplateTreeType *rnode_left_child = rnode->left();
    TemplateTreeType *rnode_right_child = rnode->right();

    // Recurse to the left and the right.
    PreProcessReferenceTree_(rnode_left_child);
    PreProcessReferenceTree_(rnode_right_child);

    // Build the node stat by combining those owned by the children.
    typename DistributedProblemType::StatisticType &rnode_left_child_stat =
      rnode_left_child->stat();
    typename DistributedProblemType::StatisticType &rnode_right_child_stat =
      rnode_right_child->stat();
    rnode_stat.Init(
      rnode_it, rnode_left_child_stat, rnode_right_child_stat);
  }
}

template<typename DistributedProblemType>
template<typename TemplateTreeType>
void core::gnp::DistributedDualtreeDfs<DistributedProblemType>::PreProcess_(
  TemplateTreeType *qnode) {

  typename DistributedProblemType::StatisticType &qnode_stat = qnode->stat();
  qnode_stat.SetZero();

  if(! qnode->is_leaf()) {
    PreProcess_(qnode->left());
    PreProcess_(qnode->right());
  }
}

#endif
