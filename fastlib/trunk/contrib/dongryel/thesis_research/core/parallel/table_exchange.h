/** @file table_exchange.h
 *
 *  A class to do a set of all-to-some table exchanges asynchronously.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_TABLE_EXCHANGE_H
#define CORE_PARALLEL_TABLE_EXCHANGE_H

#include <boost/intrusive_ptr.hpp>
#include <boost/mpi.hpp>
#include <boost/scoped_array.hpp>
#include <map>
#include "core/parallel/distributed_dualtree_task_queue.h"
#include "core/parallel/message_tag.h"
#include "core/parallel/route_request.h"
#include "core/parallel/table_exchange_message_type.h"
#include "core/table/memory_mapped_file.h"
#include "core/table/dense_matrix.h"
#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template <
typename DistributedTableType,
         typename TaskPriorityQueueType,
         typename DistributedProblemType >
class DistributedDualtreeTaskQueue;

/** @brief A class for performing an all-to-some exchange of subtrees
 *         among MPI processes.
 */
template <
typename DistributedTableType,
         typename TaskPriorityQueueType,
         typename DistributedProblemType >
class TableExchange {
  public:

    /** @brief The table type used in the exchange process.
     */
    typedef typename DistributedTableType::TableType TableType;

    /** @brief The tree type used in the exchange process.
     */
    typedef typename TableType::TreeType TreeType;

    /** @brief The subtable type used in the exchange process.
     */
    typedef core::table::SubTable<TableType> SubTableType;

    /** @brief The subtable routing type.
     */
    typedef core::parallel::RouteRequest<SubTableType> SubTableRouteRequestType;

    /** @brief The energy routing type (for detecting termination).
     */
    typedef core::parallel::RouteRequest <
    unsigned long int > EnergyRouteRequestType;

    typedef core::parallel::DistributedDualtreeTaskList <
    DistributedTableType,
    TaskPriorityQueueType,
    DistributedProblemType > ExtraTaskListType;

    /** @brief The extra task routing type (for load balancing).
     */
    typedef core::parallel::RouteRequest < ExtraTaskListType > ExtraTaskRouteRequestType;

    /** @brief The load balance request routing type.
     */
    typedef core::parallel::RouteRequest < unsigned long int > LoadBalanceRouteRequestType;

    /** @brief The distributed task queue associated with this
     *         exchange mechanism.
     */
    typedef core::parallel::DistributedDualtreeTaskQueue <
    DistributedTableType,
    TaskPriorityQueueType,
    DistributedProblemType > TaskQueueType;

    /** @brief The message type used in the recursive doubling scheme.
     */
    typedef core::parallel::MessageType <
    EnergyRouteRequestType,
    LoadBalanceRouteRequestType,
    SubTableRouteRequestType  > MessageType;

  private:

    class BatchSendReceive {
      private:

        boost::scoped_array< MessageType * > array_;

        unsigned int num_aliases_;

        void operator=(const BatchSendReceive &copy_in) {
        }

      public:

        MessageType &operator[](int pos) {
          return *(array_[pos]);
        }

        void PrepareReceive(
          unsigned int neighbor, bool do_load_balancing_in) {
          for(unsigned int i = 0; i < num_aliases_; i++) {
            array_[i]->subtable_route().object().Init(neighbor, false);
            if(do_load_balancing_in) {
              array_[i]->flush_route().object().Init(neighbor, false);
            }
          }
        }

        void Refresh(
          boost::mpi::communicator &world,
          unsigned int stage, bool for_sending,
          std::vector< MessageType > &message_cache_alias) {
          unsigned int neighbor = world.rank() ^(1 << stage);
          unsigned int lower_bound =
            (for_sending) ? ((world.rank() >> stage) << stage) :
            ((neighbor >> stage) << stage) ;
          for(unsigned int j = 0; j < num_aliases_; j++) {
            array_[j] = &(message_cache_alias[j + lower_bound ]);
          }
        }

        void Init(
          boost::mpi::communicator &world,
          unsigned int stage, bool for_sending,
          std::vector< MessageType > &message_cache_alias) {
          num_aliases_ = (1 << stage);
          boost::scoped_array< MessageType * > tmp_array(
            new  MessageType * [ num_aliases_ ]);
          array_.swap(tmp_array);
          this->Refresh(
            world, stage, for_sending, message_cache_alias);
        }

        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) {
          ar & num_aliases_;
          for(unsigned int i = 0; i < num_aliases_; i++) {
            ar & (*(array_[i]));
          }
        }
    };

    /** @brief The alias used to merge multiple MPI receive calls into
     *         one.
     */
    boost::scoped_array < BatchSendReceive > alias_receive_slots_;

    /** @brief The alias used to merge multiple MPI send calls into
     *         one.
     */
    boost::scoped_array < BatchSendReceive > alias_send_slots_;

    /** @brief Whether to do load balancing.
     */
    bool do_load_balancing_;

    /** @brief Whether to try load balancing has been tried in this
     *         stage.
     */
    bool try_load_balancing_;

    /** @brief Whether the process can enter the current recursive
     *         doubling stage.
     */
    bool enter_stage_;

    /** @brief For query subtables received from other processes and
     *         associated reference subtables with them, we store them
     *         in the extra receive slots beyond [0, world.size() ).
     */
    std::vector<int> extra_receive_slots_;

    /** @brief The MPI process needs to start checking whether this
     *         index of the cache is free before the current exchange
     *         can begin.
     */
    int last_fail_index_;

    /** @brief The pointer to the reference distributed table that is
     *         partcipating in the exchange.
     */
    DistributedTableType *reference_table_;

    /** @brief The maximum number of stages, typically log_2 of the
     *         number of MPI processes.
     */
    unsigned int max_stage_;

    /** @brief The boolean flag per each MPI prcess signifying it
     *         needs load balancing.
     */
    std::vector< unsigned long int > needs_load_balancing_;

    /** @brief The queued-up termination messages held by the current
     *         MPI process.
     */
    std::vector<EnergyRouteRequestType> queued_up_completed_computation_;

    /** @brief The queued-up extrinsic prune information that should
     *         be sent out from this MPI process.
     */
    std::vector<EnergyRouteRequestType> queued_up_extrinsic_prunes_;

    /** @brief The queued-up query subtables to be flushed for each
     *         stage.
     */
    std::vector <
    boost::intrusive_ptr < SubTableType > > queued_up_query_subtables_;

    /** @brief The current stage in the exchange process.
     */
    unsigned int stage_;

    /** @brief The messages involved in the exchange process.
     */
    std::vector < MessageType > message_cache_;

    /** @brief The number of locks applied to each cache position.
     */
    std::vector< int > message_locks_;

    /** @brief The list of cache slots that must be released after
     *         tasks are exported.
     */
    std::vector< std::pair<int, int> > post_free_cache_list_;

    /** @brief The list of cache slots that must be freed after the
     *         occupying imported query subtable is written back to
     *         its origin.
     */
    std::vector< int > post_free_flush_list_;

    /** @brief The task queue associated with the current MPI process.
     */
    TaskQueueType *task_queue_;

    /** @brief The total number of locks held by the current MPI
     *         process.
     */
    int total_num_locks_;

  private:

    void DequeueFlushRequest_(
      boost::mpi::communicator &world, int dequeue_stage) {

      message_cache_[
        world.rank()].flush_route().object().Alias(
          message_cache_[
            queued_up_query_subtables_.back()->cache_block_id()].subtable_route().object());
      message_cache_ [
        world.rank()].flush_route().set_object_is_valid_flag(true);
      message_cache_[
        world.rank()].flush_route().add_destination(
          message_cache_[
            world.rank()].flush_route().object().originating_rank());
      message_cache_[
        world.rank()].flush_route().set_stage(dequeue_stage);

      // Add to the list of flush tables to be freed.
      post_free_flush_list_.push_back(
        queued_up_query_subtables_.back()->cache_block_id());

      // Pop from the list and decrement the number of queued up
      // query subtables.
      queued_up_query_subtables_.pop_back();
    }

    void PostCleanupStage_(
      boost::mpi::communicator &world,
      unsigned int num_subtables_to_exchange,
      unsigned int lower_bound_send,
      unsigned int lower_bound_receive) {

      // For every valid send, unlock its cache.
      for(unsigned int i = 0; i < num_subtables_to_exchange; i++) {
        unsigned int send_process_rank = i + lower_bound_send;
        unsigned int receive_process_rank = i + lower_bound_receive;

        if(send_process_rank != static_cast<unsigned int>(world.rank()) &&
            message_cache_[
              send_process_rank ].subtable_route().object_is_valid()) {
          this->ReleaseCache(world, send_process_rank, 1);
        }

        if(do_load_balancing_) {
          // Free the flushed query subtables sent.
          if(message_cache_[
                send_process_rank ].flush_route().object_is_valid() &&
              message_cache_[
                send_process_rank ].flush_route().num_destinations() == 0) {
            message_cache_[
              send_process_rank ].flush_route().set_object_is_valid_flag(false);
            message_cache_[ send_process_rank ].flush_route().object().Destruct();
          }

          // Free the flushed query subtable received.
          if(message_cache_[
                receive_process_rank ].flush_route().object_is_valid() &&
              message_cache_[
                receive_process_rank ].flush_route().num_destinations() == 0) {
            message_cache_[
              receive_process_rank ].flush_route().set_object_is_valid_flag(false);
            message_cache_[
              receive_process_rank ].flush_route().object().Destruct();
          }
        }
      }

      // Free the list of flush subtables.
      for(unsigned int i = 0; i < post_free_flush_list_.size(); i++) {
        message_cache_[
          post_free_flush_list_[i] ].subtable_route().object().Destruct();
        message_cache_[
          post_free_flush_list_[i] ].subtable_route().set_object_is_valid_flag(false);
        extra_receive_slots_.push_back(post_free_flush_list_[i]);
      }
      post_free_flush_list_.resize(0);
    }

    template<typename MetricType>
    void LoadBalance_(
      boost::mpi::communicator &world, const MetricType &metric_in,
      unsigned int neighbor) {

      // If the current neighbor has less task than the self and the
      // neighbor is in need of more tasks, then donate one query
      // subtable.
      ExtraTaskListType export_list;
      ExtraTaskListType import_list;
      if(needs_load_balancing_[ neighbor ] * 4 <
          needs_load_balancing_[ world.rank()]) {
        task_queue_->PrepareExtraTaskList(
          world, metric_in, neighbor, & export_list);
        export_list.ExportPostFreeCacheList(&post_free_cache_list_);
      }
      boost::mpi::request send_request =
        world.isend(
          neighbor, core::parallel::MessageTag::EXTRA_TASK_LIST, export_list);

      while(true) {
        if(boost::optional< boost::mpi::status > l_status =
              world.iprobe(
                neighbor,
                core::parallel::MessageTag::EXTRA_TASK_LIST)) {

          world.recv(
            neighbor, core::parallel::MessageTag::EXTRA_TASK_LIST, import_list);
          break;
        }
      }
      send_request.wait();

      // Get extra task from the neighboring process if
      // available.
      import_list.Export(world, metric_in, neighbor, task_queue_);

      // Free the list of reference subtables exported to other processes.
      for(unsigned int i = 0; i < post_free_cache_list_.size(); i++) {
        this->ReleaseCache(
          world, post_free_cache_list_[i].first,
          post_free_cache_list_[i].second);
      }
      post_free_cache_list_.resize(0);
    }

    template<typename MetricType>
    void ProcessReceivedMessages_(
      boost::mpi::communicator &world,
      const MetricType &metric_in,
      unsigned int num_subtables_to_exchange,
      unsigned int neighbor,
      std::vector < boost::tuple<int, int, int, int> > *received_subtable_ids) {

      // Probe and receive the batch message.
      while(true) {
        if(boost::optional< boost::mpi::status > l_status =
              world.iprobe(
                neighbor,
                core::parallel::MessageTag::ROUTE_SUBTABLE)) {
          alias_receive_slots_[stage_].PrepareReceive(
            neighbor, do_load_balancing_);
          world.recv(
            neighbor,
            core::parallel::MessageTag::ROUTE_SUBTABLE,
            alias_receive_slots_[ stage_ ]);
          break;
        }
      }

      // Process each individual route message.
      for(unsigned int num_subtables_received = 0;
          num_subtables_received < num_subtables_to_exchange;
          num_subtables_received++) {

        // Receive the subtable.
        MessageType &tmp_route_request =
          alias_receive_slots_[stage_][num_subtables_received];
        int cache_id =
          tmp_route_request.originating_rank();
        tmp_route_request.subtable_route().object().set_cache_block_id(
          cache_id);
        if(do_load_balancing_) {
          tmp_route_request.flush_route().object().set_cache_block_id(
            cache_id);
        }

        // If this subtable is needed by the calling process, then
        // update the list of subtables received.
        MessageType &route_request = message_cache_[cache_id];

        // If the received subtable is valid,
        if(route_request.subtable_route().object_is_valid()) {

          // Lock the subtable equal to the number of remaining
          // phases.
          this->LockCache(cache_id, max_stage_ - stage_ - 1);

          // If the subtable is needed by the process, then add
          // it to its task list.
          if(route_request.subtable_route().remove_from_destination_list(world.rank())) {
            received_subtable_ids->push_back(
              boost::make_tuple(
                route_request.subtable_route().object().table()->rank(),
                route_request.subtable_route().object().start_node()->begin(),
                route_request.subtable_route().object().start_node()->count(),
                cache_id));
          }
        }
        else {
          this->ClearSubTable_(world, cache_id);
        }

        // Update the energy count.
        if(route_request.energy_route().remove_from_destination_list(world.rank()) &&
            route_request.energy_route().object_is_valid()) {
          task_queue_->decrement_remaining_global_computation(
            route_request.energy_route().object());
        }

        // Update the extrinsic prune count.
        if(route_request.extrinsic_prune_route().remove_from_destination_list(world.rank()) &&
            route_request.extrinsic_prune_route().object_is_valid()) {
          task_queue_->SeedExtrinsicPrune(
            world, route_request.extrinsic_prune_route().object());
        }

        // If load-balancing,
        if(do_load_balancing_) {

          // Synchronize with the received query subtable.
          if(route_request.flush_route().object_is_valid() &&
              route_request.flush_route().remove_from_destination_list(
                world.rank())) {
            task_queue_->Synchronize(
              world, route_request.flush_route().object());
          }
          else {
            route_request.flush_route().object().Destruct();
            route_request.flush_route().set_object_is_valid_flag(false);
          }

          // Update the computation status of other processes.
          if(
            route_request.load_balance_route().remove_from_destination_list(world.rank()) &&
            route_request.load_balance_route().object_is_valid()) {

            needs_load_balancing_[ route_request.originating_rank()] =
              route_request.load_balance_route().object();
          }

        } // end of load balancing case.

      } // end of getting all routing acknowledgements.
    }

    template<typename MetricType>
    void BufferImmediateMessage_(
      boost::mpi::communicator &world,
      const MetricType &metric_in,
      unsigned int neighbor,
      std::vector <
      std::pair <
      SubTableRouteRequestType,
      typename std::multimap<int, int>::iterator > >
      &hashed_essential_reference_subtrees_to_send,
      std::multimap<int, int> &reverse_hash_map) {

      if(message_cache_[world.rank()].subtable_route().num_destinations() == 0) {
        for(unsigned int trial = stage_; trial < max_stage_; trial++) {
          if(reverse_hash_map.count(trial) > 0) {

            // Found a subtable send request to dequeue.
            typename std::map<int, int>::iterator it = reverse_hash_map.find(trial);
            int dequeued_pos = it->second;
            SubTableRouteRequestType &route_request =
              hashed_essential_reference_subtrees_to_send[ dequeued_pos ].first;

            message_cache_[ world.rank() ].subtable_route().Init(world, route_request);
            message_cache_[ world.rank() ].subtable_route().set_object_is_valid_flag(true);
            message_cache_[ world.rank() ].subtable_route().set_stage(stage_);

            // Remove the dequeued from the hash.
            reverse_hash_map.erase(it);

            // Move the back one to the position and correct the mapping.
            if(dequeued_pos !=
                static_cast<int>(
                  hashed_essential_reference_subtrees_to_send.size()) - 1) {
              reverse_hash_map.erase(
                hashed_essential_reference_subtrees_to_send.back().second);
              hashed_essential_reference_subtrees_to_send[ dequeued_pos ] =
                hashed_essential_reference_subtrees_to_send.back();
              hashed_essential_reference_subtrees_to_send[ dequeued_pos ].second =
                reverse_hash_map.insert(
                  std::pair<int, int>(
                    hashed_essential_reference_subtrees_to_send[
                      dequeued_pos ].first.last_ready_stage(), dequeued_pos));
            }
            hashed_essential_reference_subtrees_to_send.pop_back();
            break;
          }
        }
      }

      if(do_load_balancing_) {
        // If any of the queued up flush requests is ready to be sent
        // out, then send out.
        if((! message_cache_[
              world.rank()].flush_route().object_is_valid()) &&
            queued_up_query_subtables_.size() > 0) {

          this->DequeueFlushRequest_(world, stage_);

        } // end of dequeuing flush requests.

        // Need to constantly update other processes about the current
        // progress on the self.
        if(stage_ == 0) {
          message_cache_[ world.rank()].load_balance_route().Init(world);
          message_cache_[ world.rank()].load_balance_route().add_destinations(world);
        }
        needs_load_balancing_[ world.rank()] =
          task_queue_->remaining_local_computation();
        message_cache_[ world.rank()].load_balance_route().object() =
          needs_load_balancing_[ world.rank()];
        message_cache_[ world.rank()].load_balance_route().set_object_is_valid_flag(true);
      }
    }

    void BufferInitialStageMessage_(
      boost::mpi::communicator &world,
      std::vector <
      std::pair <
      SubTableRouteRequestType,
      typename std::multimap<int, int>::iterator > >
      &hashed_essential_reference_subtrees_to_send,
      std::multimap<int, int> &reverse_hash_map) {

      // The status and the object to be copied onto.
      MessageType &new_self_send_request_object =
        message_cache_[ world.rank()];

      // Handle the overall computation messages.
      if(new_self_send_request_object.energy_route().num_destinations() == 0) {
        if(queued_up_completed_computation_.size() > 0) {

          // Examine the back of the route request list.
          EnergyRouteRequestType &route_request =
            queued_up_completed_computation_.back();

          // Prepare the initial subtable to send.
          new_self_send_request_object.energy_route().Init(
            world, route_request);
          new_self_send_request_object.energy_route().set_object_is_valid_flag(true);
          new_self_send_request_object.energy_route().set_stage(stage_);

          // Pop it from the route request list.
          queued_up_completed_computation_.pop_back();
        }
      }

      if(new_self_send_request_object.extrinsic_prune_route().num_destinations() == 0) {
        // Handle the extrinsic prune messages.
        if(queued_up_extrinsic_prunes_.size() > 0) {

          // Examine the back of the route request list.
          EnergyRouteRequestType &route_request =
            queued_up_extrinsic_prunes_.back();

          // Prepare the initial subtable to send.
          new_self_send_request_object.extrinsic_prune_route().Init(
            world, route_request);
          new_self_send_request_object.extrinsic_prune_route().set_object_is_valid_flag(true);
          new_self_send_request_object.extrinsic_prune_route().set_stage(stage_);

          // Pop it from the route request list.
          queued_up_extrinsic_prunes_.pop_back();
        }
      }

      // Set the originating rank of the message.
      new_self_send_request_object.set_originating_rank(world.rank());
    }

    /** @brief Tests whether the current MPI process can enter the
     *         given stage of the recursive-doubling exchange process.
     */
    template<typename MetricType>
    bool ReadyForStage_(
      const MetricType &metric_in, boost::mpi::communicator &world) {

      // Find out the neighbor of the next stage.
      unsigned int num_test = 1 << stage_;
      unsigned int previous_stage =
        (stage_ + (max_stage_ - 1)) % max_stage_;
      unsigned int previous_neighbor = world.rank() ^(1 << previous_stage);
      unsigned int neighbor = world.rank() ^ num_test;
      unsigned int test_lower_bound_for_receive =
        (neighbor >> stage_) << stage_;

      // Load balance if necessary.
      if(do_load_balancing_ && try_load_balancing_) {
        LoadBalance_(world, metric_in, previous_neighbor);
        try_load_balancing_ = false;
      }

      // If there are non-flushed imported query subtables, wait they
      // are out of the active status.
      if(task_queue_->num_imported_query_subtables() != 0) {
        return false;
      }

      // Whether to flush before entering the current stage.
      bool flush_only_mode =
        (task_queue_->num_exported_query_subtables() > 0 ||
         queued_up_query_subtables_.size() > 0);

      // If there is a queued up query subtables, post an asynchronous
      // send.
      if(flush_only_mode) {
        bool query_subtable_flushed = false;
        boost::mpi::request flush_request;
        if(queued_up_query_subtables_.size() > 0) {
          this->DequeueFlushRequest_(world, previous_stage);
          message_cache_[
            world.rank()].flush_route().next_destination(world);
          flush_request =
            world.isend(
              previous_neighbor, core::parallel::MessageTag::FLUSH_SUBTABLE,
              message_cache_[ world.rank()].flush_route());
          query_subtable_flushed = true;
        }

        // If there is a flush route receive posted, then do a blocking
        // receive. Otherwise, wait until the posting comes on.
        if(task_queue_->num_exported_query_subtables() > 0) {
          while(true) {
            if(boost::optional< boost::mpi::status > l_status =
                  world.iprobe(
                    previous_neighbor,
                    core::parallel::MessageTag::FLUSH_SUBTABLE))  {
              SubTableRouteRequestType received_flush_route;
              received_flush_route.object().Init(previous_neighbor, false);
              world.recv(
                previous_neighbor, core::parallel::MessageTag::FLUSH_SUBTABLE,
                received_flush_route);

              // Synchronize with the received subtable.
              task_queue_->Synchronize(world, received_flush_route.object());
              break;
            }
          }
        }

        // Wait until the flush is completely sent.
        if(query_subtable_flushed) {
          flush_request.wait();
          message_cache_[
            world.rank()].flush_route().set_object_is_valid_flag(false);
        }
        return false;
      }

      // Now, check that all of the receive buffers on this side are
      // empty.
      bool ready_flag = true;
      for(unsigned int i = last_fail_index_; ready_flag && i < num_test; i++) {
        ready_flag = (message_locks_[test_lower_bound_for_receive + i] == 0);
        if(! ready_flag) {
          last_fail_index_ = i;
        }
      }

      // Reset the test index on the self, if passing.
      if(ready_flag) {
        last_fail_index_ = 0;
        try_load_balancing_ = true;
      }

      return ready_flag ;
    }

    /** @brief Evicts an extra subtable received during load
     *         balancing.
     */
    void EvictSubTable_(boost::mpi::communicator &world, int cache_id) {
      if(message_locks_[cache_id] == 0) {
        this->ClearSubTable_(world, cache_id);
      }
      extra_receive_slots_.push_back(cache_id);
    }

    void ClearSubTable_(boost::mpi::communicator &world, int cache_id) {
      message_cache_[ cache_id ].subtable_route().object().Destruct();
      message_cache_[
        cache_id ].subtable_route().set_object_is_valid_flag(false);
    }

  public:

    /** @brief Pushes extrinsic prune information as a message to be
     *         routed.
     */
    void push_extrinsic_prunes(
      boost::mpi::communicator &world, int destination_rank_in,
      unsigned long int quantity_in) {

      if(quantity_in > 0) {
        EnergyRouteRequestType new_route_request;
        new_route_request.Init(world);
        new_route_request.set_object_is_valid_flag(true);
        new_route_request.object() = quantity_in;
        new_route_request.add_destination(destination_rank_in);
        queued_up_extrinsic_prunes_.push_back(new_route_request);
      }
    }

    /** @brief Prints the existing subtables in the cache.
     */
    void PrintSubTables(boost::mpi::communicator &world) const {
      printf("Process %d owns the subtables: with %d total locks on stage %d\n",
             world.rank(), total_num_locks_, stage_);
      printf("  Checking %d against %d\n",
             static_cast<int>(message_cache_.size()),
             static_cast<int>(message_locks_.size()));
      int total_check = 0;
      for(unsigned int i = 0; i < message_cache_.size(); i++) {
        if(message_cache_[i].subtable_route().object_is_valid()) {
          printf(
            "%d %d %d at %d locked %d times.\n",
            message_cache_[i].subtable_route().object().table()->rank(),
            message_cache_[i].subtable_route().object().table()->get_tree()->begin(),
            message_cache_[i].subtable_route().object().table()->get_tree()->count(),
            i,  message_locks_[i]);
          total_check += message_locks_[i];
        }
      }
      if(total_check != total_num_locks_) {
        printf("Got %d total locks against %d...\n", total_check,
               total_num_locks_);
      }

      printf("Queued up the following query subtables to be flushed:\n");
      for(unsigned int j = 0;
          j < queued_up_query_subtables_.size(); j++) {
        printf(
          " (%d, %d, %d) ",
          queued_up_query_subtables_[j]->subtable_id().get<0>(),
          queued_up_query_subtables_[j]->subtable_id().get<1>(),
          queued_up_query_subtables_[j]->subtable_id().get<2>());
      }
      printf("\n");
    }

    /** @brief Pushes subtables received from load-balancing.
     */
    int push_subtable(
      boost::mpi::communicator &world,
      SubTableType &subtable_in, int num_referenced_as_reference_set) {
      int receive_slot;
      if(extra_receive_slots_.size() > 0) {
        receive_slot = extra_receive_slots_.back();
        extra_receive_slots_.pop_back();
      }
      else {
        MessageType *prev_address = &(message_cache_[0]);
        message_cache_.resize(message_cache_.size() + 1);
        bool address_changed = (prev_address != (& (message_cache_[0])));
        if(address_changed) {
          for(unsigned int i = 0; i < max_stage_; i++) {
            alias_receive_slots_[i].Refresh(world, i, false, message_cache_);
            alias_send_slots_[i].Refresh(world, i, true, message_cache_);
          }
        }
        message_locks_.resize(message_locks_.size() + 1);
        receive_slot = message_cache_.size() - 1;
      }
      message_locks_[ receive_slot ] = 0;

      // Steal the pointer owned by the incoming subtable.
      message_cache_[receive_slot].subtable_route().object() = subtable_in;
      message_cache_[receive_slot].subtable_route().set_object_is_valid_flag(true);
      message_cache_[
        receive_slot].subtable_route().object().set_cache_block_id(receive_slot);
      message_cache_[
        receive_slot ].flush_route().set_object_is_valid_flag(false);
      message_cache_[
        receive_slot ].set_do_load_balancing_flag(do_load_balancing_);

      // Lock the cache.
      this->LockCache(receive_slot, num_referenced_as_reference_set);
      return receive_slot;
    }

    /** @brief Queues the query subtable and its result flush request.
     */
    void QueueFlushRequest(
      boost::mpi::communicator &world,
      const boost::intrusive_ptr<SubTableType > &query_subtable_in) {
      queued_up_query_subtables_.push_back(query_subtable_in);
    }

    /** @brief Used for prioritizing tasks, favoring subtables that
     *         arrive earlier in the exchange process.
     */
    unsigned int process_rank(
      boost::mpi::communicator &world, unsigned int test_process_rank) const {

      // Start from the most significant bit of the given process rank
      // and find the highest differing index.
      unsigned int most_significant_bit_pos = max_stage_;
      unsigned int differing_index = most_significant_bit_pos;
      for(int test_index = static_cast<int>(most_significant_bit_pos);
          test_index >= 0; test_index--) {
        unsigned int mask = 1 << test_index;
        unsigned int test_process_rank_bit = test_process_rank & mask;
        unsigned int world_rank_bit = world.rank() & mask;
        if(test_process_rank_bit != world_rank_bit) {
          differing_index = test_index;
          break;
        }
      }
      return differing_index;
    }

    /** @brief Returns the associated reference table involved in the
     *         exchange.
     */
    DistributedTableType *reference_table() {
      return reference_table_;
    }

    /** @brief Returns the load balancing flag.
     */
    bool do_load_balancing() const {
      return do_load_balancing_;
    }

    /** @brief Returns whether the current MPI process can terminate.
     */
    bool can_terminate() const {

      // Terminate when there are no queued up messages.
      return queued_up_completed_computation_.size() == 0 &&
             queued_up_extrinsic_prunes_.size() == 0 &&
             queued_up_query_subtables_.size() == 0 && stage_ == 0;
    }

    /** @brief Pushes the completed computation amount to be
     *         broadcasted to very process.
     */
    void push_completed_computation(
      boost::mpi::communicator &comm, unsigned long int quantity_in) {

      // Queue up a route message so that it can be passed to all the
      // other processes.
      if(comm.size() > 1) {
        if(queued_up_completed_computation_.size() == 0) {
          EnergyRouteRequestType new_route_request;
          new_route_request.Init(comm);
          new_route_request.set_object_is_valid_flag(true);
          new_route_request.object() = quantity_in;
          new_route_request.add_destinations(comm);
          queued_up_completed_computation_.push_back(new_route_request);
        }
        else {
          queued_up_completed_computation_.back().object() += quantity_in;
        }
      }
    }

    /** @brief The default constructor.
     */
    TableExchange() {
      do_load_balancing_ = false;
      enter_stage_ = true;
      last_fail_index_ = 0;
      max_stage_ = 0;
      reference_table_ = NULL;
      stage_ = 0;
      task_queue_ = NULL;
      total_num_locks_ = 0;
    }

    /** @brief Finds the local reference node by its ID.
     */
    TreeType *FindByBeginCount(int begin_in, int count_in) {
      return reference_table_->local_table()->get_tree()->FindByBeginCount(begin_in, count_in);
    }

    void LockCache(int cache_id, int num_times) {
      if(cache_id >= 0) {
        message_locks_[ cache_id ] += num_times;
        total_num_locks_ += num_times;
      }
    }

    void ReleaseCache(
      boost::mpi::communicator &world, int cache_id, int num_times) {
      if(cache_id >= 0 && cache_id != world.rank()) {
        message_locks_[ cache_id ] -= num_times;
        total_num_locks_ -= num_times;

        // If the subtable is not needed, free it.
        if(message_locks_[ cache_id ] == 0 &&
            message_cache_[ cache_id ].subtable_route().object_is_valid() &&
            cache_id != reference_table_->local_table()->rank()) {

          if(cache_id < world.size()) {

            // Clear the subtable.
            this->ClearSubTable_(world, cache_id);
          }
          else if(! message_cache_[
                    cache_id ].subtable_route().object().is_query_subtable()) {

            // If it is among the extra subtables, then evict it.
            this->EvictSubTable_(world, cache_id);
          }
        }
      }
    }

    /** @brief Grabs the subtable in the given cache position.
     */
    SubTableType *FindSubTable(int cache_id) {
      SubTableType *returned_subtable = NULL;
      if(cache_id >= 0) {
        returned_subtable =
          &(message_cache_[cache_id].subtable_route().object());
      }
      return returned_subtable;
    }

    /** @brief Signals that the load balancing is necessary on this
     *         process.
     */
    void turn_on_load_balancing(
      boost::mpi::communicator &world,
      unsigned long int remaining_local_computation_in) {
      needs_load_balancing_[ world.rank()] = remaining_local_computation_in;
    }

    /** @brief Initialize the all-to-some exchange object with a
     *         distributed table and the cache size.
     */
    void Init(
      boost::mpi::communicator &world,
      int max_subtree_size_in,
      bool do_load_balancing_in,
      DistributedTableType *query_table_in,
      DistributedTableType *reference_table_in,
      TaskQueueType *task_queue_in) {

      // Load balancing option.
      needs_load_balancing_.resize(world.size());
      for(int i = 0; i < world.size(); i++) {
        needs_load_balancing_[i] = 0;
      }
      do_load_balancing_ = do_load_balancing_in;
      try_load_balancing_ = true;

      // Set the pointer to the task queue.
      task_queue_ = task_queue_in;

      // Initialize the stages.
      stage_ = 0;
      enter_stage_ = true;

      // The maximum number of neighbors.
      max_stage_ = core::math::RoundLogBaseTwo(world.size());

      // Set the reference distributed table.
      reference_table_ = reference_table_in;

      // Preallocate the cache.
      message_cache_.resize(world.size());
      for(int i = 0; i < world.size(); i++) {
        message_cache_[i].set_do_load_balancing_flag(do_load_balancing_);
      }
      extra_receive_slots_.resize(0);

      // Set up the alias slots.
      boost::scoped_array< BatchSendReceive > tmp_receive_slots(
        new BatchSendReceive[max_stage_]);
      alias_receive_slots_.swap(tmp_receive_slots);
      boost::scoped_array< BatchSendReceive > tmp_send_slots(
        new BatchSendReceive[max_stage_]);
      alias_send_slots_.swap(tmp_send_slots);
      for(unsigned int i = 0; i < max_stage_; i++) {
        alias_receive_slots_[i].Init(world, i, false, message_cache_);
        alias_send_slots_[i].Init(world, i, true, message_cache_);
      }

      // Initialize the queues.
      queued_up_completed_computation_.resize(0);
      queued_up_extrinsic_prunes_.resize(0);

      // Initialize the locks.
      message_locks_.resize(message_cache_.size());
      std::fill(
        message_locks_.begin(), message_locks_.end(), 0);
      total_num_locks_ = 0;
    }

    /** @brief Issue a set of asynchronous send and receive
     *         operations.
     *
     *  @return received_subtables The list of received subtables.
     */
    template<typename MetricType>
    void SendReceive(
      const MetricType &metric_in,
      boost::mpi::communicator &world,
      std::vector <
      std::pair <
      SubTableRouteRequestType,
      typename std::multimap<int, int>::iterator > >
      &hashed_essential_reference_subtrees_to_send,
      std::multimap<int, int> &reverse_hash_map) {

      // Proceed with the stage, if ready.
      if((! task_queue_->can_terminate(world)) && enter_stage_) {

        // The ID of the received subtables.
        std::vector< boost::tuple<int, int, int, int> > received_subtable_ids;

        // Exchange with the current neighbor.
        unsigned int num_subtables_to_exchange = (1 << stage_);
        unsigned int neighbor = world.rank() ^(1 << stage_);
        unsigned int lower_bound_send = (world.rank() >> stage_) << stage_;
        unsigned int lower_bound_receive = (neighbor >> stage_) << stage_;

        // Clear the list of received subtables in this round.
        received_subtable_ids.resize(0);

        // At the start of each phase (stage == 0), dequeue messages
        // to be sent out.
        if(stage_ == 0) {
          this->BufferInitialStageMessage_(
            world, hashed_essential_reference_subtrees_to_send,
            reverse_hash_map);
        } // end of checking whether the stage is 0.

        // Send any of the queued up messages that can be sent
        // immediately in this stage.
        {
          this->BufferImmediateMessage_(
            world, metric_in, neighbor,
            hashed_essential_reference_subtrees_to_send, reverse_hash_map);
        }

        for(unsigned int i = 0; i < num_subtables_to_exchange; i++) {
          unsigned int subtable_send_index = i + lower_bound_send;
          MessageType &send_request_object =
            message_cache_[ subtable_send_index ];
          send_request_object.next_destination(world);

          // For each subtable sent, we expect something from the neighbor.
          //message_send_request_[i] =
          //world.isend(
          //  neighbor, core::parallel::MessageTag::ROUTE_SUBTABLE,
          //  send_request_object);
        }
        boost::mpi::request message_send_request =
          world.isend(
            neighbor, core::parallel::MessageTag::ROUTE_SUBTABLE,
            alias_send_slots_[ stage_ ]);

        // Handle incoming messages from the neighbor.
        this->ProcessReceivedMessages_(
          world, metric_in, num_subtables_to_exchange,
          neighbor, &received_subtable_ids);

        // Wait until all sends are done.
        message_send_request.wait();

        // Generate more tasks.
        task_queue_->GenerateTasks(world, metric_in, received_subtable_ids);

        // Post-cleanup.
        this->PostCleanupStage_(
          world, num_subtables_to_exchange,
          lower_bound_send, lower_bound_receive);

        // Increment the stage when done, and turn off the stage flag.
        stage_ = (stage_ + 1) % max_stage_;
        enter_stage_ = false;

      } // end of entering the stage.

      // Do post-flushes before entering.
      if(! enter_stage_) {
        enter_stage_ = this->ReadyForStage_(metric_in, world);
      }
    }
};
}
}

#endif
