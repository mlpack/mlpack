/** @file reference_tree_walker.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_REFERENCE_TREE_WALKER_H
#define CORE_PARALLEL_REFERENCE_TREE_WALKER_H

#include "core/gnp/dualtree_trace.h"
#include "core/math/math_lib.h"
#include "core/parallel/route_request.h"

namespace core {
namespace parallel {

template< typename DistributedTableType, typename DistributedProblemType >
class ReferenceTreeWalker {

  public:

    /** @brief The associated serial problem type.
     */
    typedef typename DistributedProblemType::ProblemType ProblemType;

    /** @brief The table type used in the exchange process.
     */
    typedef typename DistributedTableType::TableType TableType;

    /** @brief The iterator type.
     */
    typedef typename TableType::TreeIterator TreeIteratorType;

    /** @brief The global type used in the distributed problem.
     */
    typedef typename DistributedProblemType::GlobalType GlobalType;

    /** @brief The tree type used in the exchange process.
     */
    typedef typename TableType::TreeType TreeType;

    /** @brief The subtable type used in the exchange process.
     */
    typedef core::table::SubTable<TableType> SubTableType;

    /** @brief The ID of subtables.
     */
    typedef typename SubTableType::SubTableIDType SubTableIDType;

    /** @brief The routing request type.
     */
    typedef core::parallel::RouteRequest<SubTableType> SubTableRouteRequestType;

  private:

    std::vector< unsigned long int > extrinsic_prunes_;

    DistributedTableType *query_table_;

    std::vector <
    std::vector< std::pair<int, int> > > essential_reference_subtrees_;

    TableType *local_reference_table_;

    unsigned long int max_num_reference_points_to_pack_per_process_;

    int max_reference_subtree_size_;

    std::vector< unsigned long int >
    num_reference_points_assigned_per_process_;

    core::gnp::DualtreeTrace <
    boost::tuple< TreeType *, TreeType *, core::math::Range > > trace_;

    bool weak_scaling_measuring_mode_;

  private:

    void HashSendList_(
      boost::mpi::communicator &world,
      const std::pair<int, int> &local_rnode_id,
      int query_process_id,
      std::vector <
      core::parallel::RouteRequest<SubTableType> > *
      hashed_essential_reference_subtrees) {

      // May consider using a STL map to speed up the hashing.
      int found_index = -1;
      for(int i = hashed_essential_reference_subtrees->size() - 1;
          i >= 0; i--) {
        if((*hashed_essential_reference_subtrees)[i].object().
            has_same_subtable_id(local_rnode_id)) {
          found_index = i;
          break;
        }
      }
      if(found_index < 0) {
        hashed_essential_reference_subtrees->resize(
          hashed_essential_reference_subtrees->size() + 1);
        found_index = hashed_essential_reference_subtrees->size() - 1;
        (*hashed_essential_reference_subtrees)[ found_index ].Init(world);
        (*hashed_essential_reference_subtrees)[
          found_index].object().Init(
            local_reference_table_,
            local_reference_table_->get_tree()->FindByBeginCount(
              local_rnode_id.first, local_rnode_id.second), false);
        (*hashed_essential_reference_subtrees)[
          found_index].set_object_is_valid_flag(true);
      }
      (*hashed_essential_reference_subtrees)[
        found_index].add_destination(query_process_id);
    }

  public:

    /** @brief The default constructor.
     */
    ReferenceTreeWalker() {
      query_table_ = NULL;
      local_reference_table_ = NULL;
      max_num_reference_points_to_pack_per_process_ =
        std::numeric_limits<unsigned long int>::max();
      max_reference_subtree_size_ = 0;
      weak_scaling_measuring_mode_ = false;
    }

    /** @brief Walks the local reference tree to generate more
     *         messages to send.
     */
    template < typename MetricType,
             typename DistributedDualtreeTaskQueueType >
    void Walk(
      const MetricType &metric_in,
      const GlobalType &global_in,
      boost::mpi::communicator &world,
      int max_hashed_subtrees_to_queue,
      std::vector < SubTableRouteRequestType >
      *hashed_essential_reference_subtrees_to_send,
      DistributedDualtreeTaskQueueType *distributed_tasks_in) {

      // Return if the walk is done.
      if(trace_.empty()) {
        return;
      }

      // Temporary variables.
      TreeType *first_partner = NULL, *second_partner = NULL;
      core::math::Range first_squared_distance_range;
      core::math::Range second_squared_distance_range;

      do {

        // Termination flag.
        bool should_terminate = false;

        // Push a blank argument to the trace for making the exit phase.
        if(trace_.front().get<0>() != NULL) {
          trace_.push_front(
            boost::tuple <
            TreeType *, TreeType *, core::math::Range > (
              NULL, NULL, core::math::Range()));
        }

        // Pop the next item to visit in the list.
        boost::tuple <
        TreeType *, TreeType *, core::math::Range > args = trace_.back();
        trace_.pop_back();

        while(
          trace_.empty() == false && args.get<0>() != NULL) {

          // Get the arguments.
          TreeType *global_query_node = args.get<0>();
          TreeType *local_reference_node = args.get<1>();

          // Determine if the pair can be pruned.
          if(global_in.ConsiderExtrinsicPrune(args.get<2>())) {
            typename TableType::TreeIterator qnode_it =
              query_table_->get_node_iterator(global_query_node);
            while(qnode_it.HasNext()) {
              int query_process_id;
              qnode_it.Next(&query_process_id);
              extrinsic_prunes_[query_process_id] +=
                local_reference_node->count();
            }
          } // end of prunable case.

          // Non-prunable-case.
          else {

            if(global_query_node->is_leaf()) {

              // If the reference node size is within the size limit, then for
              // each query process in the set, add the reference node to the
              // list.
              if(local_reference_node->count() <= max_reference_subtree_size_ ||
                  local_reference_node->is_leaf()) {
                typename TableType::TreeIterator qnode_it =
                  query_table_->get_node_iterator(global_query_node);
                std::pair<int, int> local_rnode_id(
                  local_reference_node->begin(), local_reference_node->count());
                while(qnode_it.HasNext()) {
                  int query_process_id;
                  qnode_it.Next(&query_process_id);

                  // Only add if, weak scaling measuring is disabled or the
                  // query process rank differs from the reference process rank
                  // by at most 2 bits.
                  if((! weak_scaling_measuring_mode_)  ||
                      (core::math::BitCount(
                         query_process_id, world.rank()) <= 2 &&
                       num_reference_points_assigned_per_process_[
                         query_process_id ] <
                       max_num_reference_points_to_pack_per_process_)) {
                    essential_reference_subtrees_[
                      query_process_id ].push_back(local_rnode_id);
                    num_reference_points_assigned_per_process_[
                      query_process_id ] += local_rnode_id.second;

                    // Add the query process ID to the list of query processes
                    // that this reference subtree needs to be sent to.
                    if(query_process_id != world.rank()) {
                      HashSendList_(
                        world, local_rnode_id, query_process_id,
                        hashed_essential_reference_subtrees_to_send);
                    }
                  } // non-prune case.

                  // Otherwise, consider it as pruned.
                  else {
                    extrinsic_prunes_ [
                      query_process_id] += local_reference_node->count();
                  }
                }
              } // qnode leaf, rnode leaf.
              else {
                core::gnp::DualtreeDfs<ProblemType>::Heuristic(
                  metric_in,
                  global_query_node,
                  local_reference_node->left(),
                  local_reference_node->right(),
                  &first_partner,
                  first_squared_distance_range,
                  &second_partner,
                  second_squared_distance_range);
                trace_.push_back(
                  boost::tuple <
                  TreeType *, TreeType *, core::math::Range > (
                    global_query_node, first_partner,
                    first_squared_distance_range));
                trace_.push_front(
                  boost::tuple <
                  TreeType *, TreeType *, core::math::Range > (
                    global_query_node, second_partner,
                    second_squared_distance_range));
              } // qnode leaf, rnode non-leaf.
            } // end of query node being a leaf.

            else {

              if(local_reference_node->count() <= max_reference_subtree_size_ ||
                  local_reference_node->is_leaf()) {
                core::gnp::DualtreeDfs<ProblemType>::Heuristic(
                  metric_in,
                  local_reference_node,
                  global_query_node->left(),
                  global_query_node->right(),
                  &first_partner,
                  first_squared_distance_range,
                  &second_partner,
                  second_squared_distance_range);
                trace_.push_back(
                  boost::tuple <
                  TreeType *, TreeType *, core::math::Range > (
                    first_partner, local_reference_node,
                    first_squared_distance_range));
                trace_.push_front(
                  boost::tuple <
                  TreeType *, TreeType *, core::math::Range > (
                    second_partner, local_reference_node,
                    second_squared_distance_range));
              } // qnode non-leaf, rnode leaf.
              else {
                core::gnp::DualtreeDfs<ProblemType>::Heuristic(
                  metric_in,
                  global_query_node->left(),
                  local_reference_node->left(),
                  local_reference_node->right(),
                  &first_partner,
                  first_squared_distance_range,
                  &second_partner,
                  second_squared_distance_range);
                trace_.push_back(
                  boost::tuple <
                  TreeType *, TreeType *, core::math::Range > (
                    global_query_node->left(), first_partner,
                    first_squared_distance_range));
                trace_.push_front(
                  boost::tuple <
                  TreeType *, TreeType *, core::math::Range > (
                    global_query_node->left(), second_partner,
                    second_squared_distance_range));

                core::gnp::DualtreeDfs<ProblemType>::Heuristic(
                  metric_in,
                  global_query_node->right(),
                  local_reference_node->left(),
                  local_reference_node->right(),
                  &first_partner,
                  first_squared_distance_range,
                  &second_partner,
                  second_squared_distance_range);
                trace_.push_back(
                  boost::tuple <
                  TreeType *, TreeType *, core::math::Range > (
                    global_query_node->right(), first_partner,
                    first_squared_distance_range));
                trace_.push_front(
                  boost::tuple <
                  TreeType *, TreeType *, core::math::Range > (
                    global_query_node->right(), second_partner,
                    second_squared_distance_range));
              } // qnode, rnode non-leaves.
            } // end of non-leaf qnode case.
          } // end of non-prunable case.

          // Pop the next item in the list, if we should continue.
          should_terminate =
            ((world.size() == 1 &&
              static_cast<int>(essential_reference_subtrees_[world.rank()].size()) >= max_hashed_subtrees_to_queue) ||
             (world.size() > 1 &&
              static_cast<int>(hashed_essential_reference_subtrees_to_send->size()) >= max_hashed_subtrees_to_queue &&
              static_cast<int>(essential_reference_subtrees_[world.rank()].size()) >= omp_get_max_threads()) ||
             trace_.empty());

          if(! should_terminate) {
            args = trace_.back();
            trace_.pop_back();
          }
          else {
            break;
          }

        } // end of the inner loop.

        // The termination condition.
        should_terminate =
          ((world.size() == 1 &&
            static_cast<int>(essential_reference_subtrees_[world.rank()].size()) >= max_hashed_subtrees_to_queue) ||
           (world.size() > 1 &&
            static_cast<int>(hashed_essential_reference_subtrees_to_send->size()) >= max_hashed_subtrees_to_queue &&
            static_cast<int>(essential_reference_subtrees_[world.rank()].size()) >= omp_get_max_threads()) ||
           trace_.empty());
        if(should_terminate) {
          break;
        }
      }
      while(true);

      // Generate tasks on the self.
      std::vector< boost::tuple<int, int, int, int> > received_subtable_ids;
      for(unsigned int j = 0;
          j < essential_reference_subtrees_[world.rank()].size(); j++) {

        int reference_begin =
          essential_reference_subtrees_[ world.rank()][j].first;
        int reference_count =
          essential_reference_subtrees_[ world.rank()][j].second;

        // Reference subtables on the self are already available.
        received_subtable_ids.push_back(
          boost::make_tuple(
            world.rank() , reference_begin, reference_count, -1));
      }

      // Fill out the initial task consisting of the reference trees on
      // the same process.
      distributed_tasks_in->GenerateTasks(
        world, metric_in, received_subtable_ids);
      essential_reference_subtrees_.resize(0);
      essential_reference_subtrees_.resize(world.size());

      // For each query process, queue up how much computation is
      // pruned.
      for(int i = 0; i < world.size(); i++) {
        unsigned long int completed_work =
          static_cast<unsigned long int>(
            query_table_->local_n_entries(i)) *
          extrinsic_prunes_[ i ];
        distributed_tasks_in->push_completed_computation(
          world, i, extrinsic_prunes_[i], completed_work);

        // Clear the extrinsic prunes.
        extrinsic_prunes_[i] = 0;
      }
    }

    /** @brief Initializes the reference tree walker.
     */
    template<typename MetricType>
    void Init(
      const MetricType &metric_in,
      boost::mpi::communicator &world,
      DistributedTableType *query_table_in,
      TableType *local_reference_table_in,
      int max_reference_subtree_size_in,
      bool weak_scaling_mode_in,
      unsigned long int max_num_reference_points_to_pack_per_process_in) {

      // Set the maximum number of reference points to pack per
      // process.
      max_num_reference_points_to_pack_per_process_ =
        max_num_reference_points_to_pack_per_process_in;

      // Set the weak scaling mode.
      weak_scaling_measuring_mode_ = weak_scaling_mode_in;

      // Set the maximum reference subtree for task grain.
      max_reference_subtree_size_ = max_reference_subtree_size_in;

      // Set the tables.
      query_table_ = query_table_in;
      local_reference_table_ = local_reference_table_in;

      // Initialize the assigned number of points assigned to zero.
      num_reference_points_assigned_per_process_.resize(world.size());
      std::fill(
        num_reference_points_assigned_per_process_.begin(),
        num_reference_points_assigned_per_process_.end(), 0);

      // Initialize the stack.
      core::math::Range squared_distance_range =
        query_table_->get_tree()->bound().RangeDistanceSq(
          metric_in, local_reference_table_->get_tree()->bound());
      trace_.Init(100000);
      trace_.push_back(
        boost::tuple <
        TreeType *, TreeType *, core::math::Range > (
          query_table_->get_tree(), local_reference_table_->get_tree(),
          squared_distance_range));

      // Initialize the extrinsic prune counts.
      extrinsic_prunes_.resize(world.size());
      std::fill(extrinsic_prunes_.begin(), extrinsic_prunes_.end(), 0);

      // Essential reference subtrees.
      essential_reference_subtrees_.resize(world.size());
    }
};
}
}

#endif
