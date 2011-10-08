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

template< typename DistributedTableType >
class ReferenceTreeWalker {

  public:

    /** @brief The table type used in the exchange process.
     */
    typedef typename DistributedTableType::TableType TableType;

    /** @brief The iterator type.
     */
    typedef typename TableType::TreeIterator TreeIteratorType;

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

    int max_reference_subtree_size_;

    std::vector< unsigned long int >
    num_reference_points_assigned_per_process_;

    core::gnp::DualtreeTace< std::pair<TreeType *, TreeType *> > trace_;

    bool weak_scaling_measuring_mode_;

  private:

    void HashSendList_(
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
        (*hashed_essential_reference_subtrees)[ found_index ].Init(*world_);
        (*hashed_essential_reference_subtrees)[
          found_index].object().Init(
            reference_table_->local_table(),
            reference_table_->local_table()->get_tree()->FindByBeginCount(
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
      max_reference_subtree_size_ = 0;
      weak_scaling_measuring_mode_ = false;
    }

    /** @brief Walks the local reference tree to generate more
     *         messages to send.
     */
    template < typename MetricType, typename GlobalType,
             typename DistributedDualtreeTaskQueueType >
    void Walk(
      const MetricType &metric_in,
      const GlobalType &global_in,
      boost::mpi::communicator &world,
      int max_hashed_subtrees_to_queue,
      TreeType *global_qnode,
      std::vector < SubTableRouteRequestType >
      *hashed_essential_reference_subtrees_to_send,
      DistributedDualtreeTaskQueueType *distributed_tasks_in) {

      // Push a blank argument to the trace for making the exit phase.
      trace_.push_front(std::pair<TreeType *, TreeType *>(NULL, NULL));

      // Pop the next item to visit in the list.
      std::pair<TreeType *, TreeType *> args = trace_.back();
      trace_.pop_back();

      do {
        while(
          trace_.empty() == false && args.second != NULL &&
          hashed_essential_reference_subtrees_to_send->size() <
          max_hashed_subtrees_to_queue) {

          // Get the arguments.
          TreeType *global_query_node = args.first;
          TreeType *local_reference_node = args.second;
          core::math::Range squared_distance_range =
            global_query_node->bound().RangeDistanceSq(
              metric_in, local_reference_node->bound());

          // Determine if the pair can be pruned.
          if(global().ConsiderExtrinsicPrune(squared_distance_range)) {
            typename TableType::TreeIterator qnode_it =
              query_table_->get_node_iterator(global_query_node);
            while(qnode_it.HasNext()) {
              int query_process_id;
              qnode_it.Next(&query_process_id);
              extrinsic_prunes_[query_process_id] +=
                local_reference_node->count();
            }
            continue;
          }

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
                       query_process_id, world_->rank()) <= 2 &&
                     num_reference_points_assigned_per_process_[
                       query_process_id ] <
                     max_num_reference_points_to_pack_per_process_)) {
                  essential_reference_subtrees_[
                    query_process_id ].push_back(local_rnode_id);
                  num_reference_points_assigned_per_process_[
                    query_process_id ] += local_rnode_id.second;

                  // Add the query process ID to the list of query processes
                  // that this reference subtree needs to be sent to.
                  if(query_process_id != world_->rank()) {
                    HashSendList_(
                      local_rnode_id, query_process_id,
                      hashed_essential_reference_subtrees);
                  }
                }

                // Otherwise, consider it as pruned.
                else {
                  extrinsic_prunes_ [
                    query_process_id] += local_reference_node->count();
                }
              }
            }
            else {
              trace_.push_back(
                std::pair <
                TreeType *, TreeType * > (
                  global_query_node, local_reference_node->left()));
              trace_.push_front(
                std::pair <
                TreeType *, TreeType * > (
                  global_query_node, local_reference_node->right()));
            }
            continue;
          }

          // Here, we know that the global query node is a non-leaf, so we
          // need to split both ways.
          if(local_reference_node->count() <= max_reference_subtree_size_ ||
              local_reference_node->is_leaf()) {
            trace_.push_back(
              std::pair <
              TreeType *, TreeType * > (
                global_query_node->left(), local_reference_node));
            trace_.push_front(
              std::pair <
              TreeType *, TreeType * > (
                global_query_node->right(), local_reference_node));
          }
          else {
            trace_.push_back(
              std::pair <
              TreeType *, TreeType * > (
                global_query_node->left(), local_reference_node->left()));
            trace_.push_front(
              std::pair <
              TreeType *, TreeType * > (
                global_query_node->left(), local_reference_node->right()));
            trace_.push_back(
              std::pair <
              TreeType *, TreeType * > (
                global_query_node->right(), local_reference_node->left()));
            trace_.push_front(
              std::pair <
              TreeType *, TreeType * > (
                global_query_node->right(), local_reference_node->right()));
          }
        } // end of the inner loop.

        // The termination condition.
        if(trace_.empty() ||
            hashed_essential_reference_subtrees_to_send->size() >=
            max_hashed_subtrees_to_queue) {
          break;
        }
      }
      while(true);

      // Generate tasks on the self.
      std::vector< boost::tuple<int, int, int, int> > received_subtable_ids;
      for(unsigned int j = 0;
          j < essential_reference_subtrees_to_send_[world.rank()].size(); j++) {

        int reference_begin =
          essential_reference_subtrees_to_send_[ world.rank()][j].first;
        int reference_count =
          essential_reference_subtrees_to_send_[ world.rank()][j].second;

        // Reference subtables on the self are already available.
        received_subtable_ids.push_back(
          boost::make_tuple(i, reference_begin, reference_count, -1));
      }

      // Fill out the initial task consisting of the reference trees on
      // the same process.
      distributed_tasks_in->GenerateTasks(
        world, metric, received_subtable_ids);
      essential_reference_subtrees_to_send_.resize(0);
      essential_reference_subtrees_to_send_.resize(world.size());


      // Queue prune counts.
    }

    /** @brief Initializes the reference tree walker.
     */
    void Init(
      boost::mpi::communicator &world,
      DistributedTableType *query_table_in,
      TableType *local_reference_table_in,
      int max_reference_subtree_size_in,
      bool weak_scaling_mode_in) {

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
      trace_.Init();
      trace_.push_back(
        std::pair <
        TreeType *, TreeType * > (
          query_table_->get_tree(), local_reference_table_->get_tree()));

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
