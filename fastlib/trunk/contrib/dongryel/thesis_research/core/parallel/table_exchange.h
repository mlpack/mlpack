/** @file table_exchange.h
 *
 *  A class to do a set of all-to-some table exchanges asynchronously.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_TABLE_EXCHANGE_H
#define CORE_PARALLEL_TABLE_EXCHANGE_H

#include <boost/mpi.hpp>
#include "core/parallel/distributed_dualtree_task_queue.h"
#include "core/parallel/message_tag.h"
#include "core/parallel/route_request.h"
#include "core/table/memory_mapped_file.h"
#include "core/table/dense_matrix.h"
#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template <
typename DistributedTableType,
         typename TaskPriorityQueueType,
         typename QueryResultType >
class DistributedDualtreeTaskQueue;

/** @brief A class for performing an all-to-some exchange of subtrees
 *         among MPI processes.
 */
template <
typename DistributedTableType,
         typename TaskPriorityQueueType,
         typename QueryResultType >
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

    typedef core::parallel::RouteRequest<SubTableType> SubTableRouteRequestType;

    typedef core::parallel::RouteRequest< unsigned long int > EnergyRouteRequestType;

    typedef core::parallel::RouteRequest <
    std::pair<SubTableType, QueryResultType> > QuerySubTableFlushRequestType;

    typedef core::parallel::DistributedDualtreeTaskQueue <
    DistributedTableType, TaskPriorityQueueType, QueryResultType > TaskQueueType;

    class MessageType {
      private:

        // For serialization.
        friend class boost::serialization::access;

      private:
        int originating_rank_;

        SubTableRouteRequestType subtable_route_;

        EnergyRouteRequestType energy_route_;

        QuerySubTableFlushRequestType flush_route_;

      public:

        MessageType() {
          originating_rank_ = 0;
        }

        void operator=(const MessageType &message_in) {
          originating_rank_ = message_in.originating_rank();
          subtable_route_ = message_in.subtable_route();
          energy_route_ = message_in.energy_route();
          flush_route_ = message_in.flush_route();
        }

        MessageType(const MessageType &message_in) {
          this->operator=(message_in);
        }

        int next_destination(boost::mpi::communicator &comm) {
          subtable_route_.next_destination(comm);
          energy_route_.next_destination(comm);
          return flush_route_.next_destination(comm);
        }

        void set_originating_rank(int rank_in) {
          originating_rank_ = rank_in;
        }

        int originating_rank() const {
          return originating_rank_;
        }

        QuerySubTableFlushRequestType &flush_route() {
          return flush_route_;
        }

        const QuerySubTableFlushRequestType &flush_route() const {
          return flush_route_;
        }

        SubTableRouteRequestType &subtable_route() {
          return subtable_route_;
        }

        const SubTableRouteRequestType &subtable_route() const {
          return subtable_route_;
        }

        EnergyRouteRequestType &energy_route() {
          return energy_route_;
        }

        const EnergyRouteRequestType &energy_route() const {
          return energy_route_;
        }

        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) {
          ar & originating_rank_;
          ar & subtable_route_;
          ar & energy_route_;
          ar & flush_route_;
        }
    };

  private:

    /** @brief Whether the MPI process can enter the current exchange
     *         phase.
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

    /** @brief The pointer to the local table that is partcipating in
     *         the exchange.
     */
    TableType *local_table_;

    /** @brief The maximum number of stages, typically log_2 of the
     *         number of MPI processes.
     */
    unsigned int max_stage_;

    /** @brief The queued-up termination messages held by the current
     *         MPI process.
     */
    std::vector<EnergyRouteRequestType> queued_up_completed_computation_;

    /** @brief The current stage in the exchange process.
     */
    unsigned int stage_;

    /** @brief The messages involved in the exchange process.
     */
    std::vector < MessageType > message_cache_;

    /** @brief The list of MPI requests that are tested for sending.
     */
    std::vector< boost::mpi::request > message_send_request_;

    /** @brief The number of locks applied to each cache position.
     */
    std::vector< int > message_locks_;

    /** @brief Used for limiting the number of points cached in the
     *         current MPI process.
     */
    unsigned long int remaining_extra_points_to_hold_;

    /** @brief The task queue associated with the current MPI process.
     */
    TaskQueueType *task_queue_;

    /** @brief The total number of locks held by the current MPI
     *         process.
     */
    int total_num_locks_;

  private:

    bool ReadyForStage_(boost::mpi::communicator &world) {

      // Find out the neighbor of the next stage.
      unsigned int num_test = 1 << stage_;
      unsigned int neighbor = world.rank() ^ num_test;
      unsigned int test_lower_bound = (neighbor >> stage_) << stage_;

      // Now, check that all the other receive buffers are empty.
      bool ready_flag = true;
      for(unsigned int i = last_fail_index_; ready_flag && i < num_test; i++) {
        ready_flag = (message_locks_[test_lower_bound + i] == 0);
        if(! ready_flag) {
          last_fail_index_ = i;
        }
      }
      enter_stage_ = ready_flag;
      if(enter_stage_) {
        last_fail_index_ = 0;
      }
      return ready_flag;
    }

    void LoadBalance_(boost::mpi::communicator & world, int neighbor) {

      // Send to the neighbor what the status is on the current MPI
      // process.
      boost::mpi::request send_request;
      core::parallel::DualtreeLoadBalanceRequest <
      SubTableType > *load_balance_request = NULL;
      task_queue_->SendLoadBalanceRequest(
        world, neighbor, &load_balance_request, &send_request);

      // Wait until the message from the neighbor is received.
      core::parallel::DualtreeLoadBalanceRequest <
      SubTableType > neighbor_load_balance_request;
      while(true) {
        if(boost::optional< boost::mpi::status > l_status =
              world.iprobe(
                neighbor,
                core::parallel::MessageTag::LOAD_BALANCE_REQUEST)) {
          world.recv(
            neighbor, core::parallel::MessageTag::LOAD_BALANCE_REQUEST,
            neighbor_load_balance_request);
          break;
        }
      }

      // Wait until the send request is completed.
      send_request.wait();
      delete load_balance_request;

      // Now prepare the task list that must be sent to the neighbor.
      if(neighbor_load_balance_request.needs_load_balancing()) {

      }

      // Asynchronously send the extra work to the neighbor.
      //boost::mpi::request task_list_send =
      //world.isend();

      // Receive from the neighbor.
      while(true) {
        if(boost::optional< boost::mpi::status > l_status =
              world.iprobe(
                neighbor,
                core::parallel::MessageTag::TASK_LIST)) {
          world.recv(
            neighbor, core::parallel::MessageTag::TASK_LIST,
            neighbor_load_balance_request);
          break;
        }
      }

      // Wait until the send request is completed.
      //task_list_send.wait();
    }

    /** @brief Prints the existing subtables in the cache.
     */
    void PrintSubTables_(boost::mpi::communicator &world) {
      printf("\n\nProcess %d owns the subtables:\n", world.rank());
      for(unsigned int i = 0; i < message_cache_.size(); i++) {
        printf(
          "%d %d %d\n",
          message_cache_[i].subtable_route().table()->rank(),
          message_cache_[i].subtable_route().table()->get_tree()->begin(),
          message_cache_[i].subtable_route().table()->get_tree()->count());
      }
    }

    /** @brief Evicts an extra subtable received during load
     *         balancing.
     */
    void EvictSubTable_(int cache_id) {
      this->ClearSubTable_(cache_id);
      extra_receive_slots_.push_back(cache_id);
    }

    void ClearSubTable_(int cache_id) {
      message_cache_[ cache_id ].subtable_route().object().Destruct();
    }

  public:

    /** @brief Queues the query subtable and its result flush request.
     */
    void QueueFlushRequest(
      SubTableType &query_subtable_in, QueryResultType &query_result_in,
      int destination_rank_in) {


    }

    unsigned long int remaining_extra_points_to_hold() const {
      return remaining_extra_points_to_hold_;
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

    TableType *local_table() {
      return local_table_;
    }

    bool can_terminate() const {
      return queued_up_completed_computation_.size() == 0 && stage_ == 0;
    }

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
      enter_stage_ = true;
      last_fail_index_ = 0;
      local_table_ = NULL;
      max_stage_ = 0;
      remaining_extra_points_to_hold_ = 0;
      stage_ = 0;
      task_queue_ = NULL;
      total_num_locks_ = 0;
    }

    TreeType *FindByBeginCount(int begin_in, int count_in) {
      return local_table_->get_tree()->FindByBeginCount(begin_in, count_in);
    }

    void LockCache(int cache_id, int num_times) {
      if(cache_id >= 0) {
        message_locks_[ cache_id ] += num_times;
        total_num_locks_ += num_times;
      }
    }

    void ReleaseCache(
      boost::mpi::communicator &world, int cache_id, int num_times) {
      if(cache_id >= 0) {
        message_locks_[ cache_id ] -= num_times;
        total_num_locks_ -= num_times;

        // If the subtable is not needed, free it.
        if(message_locks_[ cache_id ] == 0 &&
            message_cache_[ cache_id ].subtable_route().object_is_valid() &&
            cache_id != local_table_->rank()) {

          if(cache_id < world.size()) {

            // Clear the subtable.
            this->ClearSubTable_(cache_id);
          }
          else {

            // If it is among the extra subtables, then evict it.
            this->EvictSubTable_(cache_id);
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

    /** @brief Initialize the all-to-some exchange object with a
     *         distributed table and the cache size.
     */
    void Init(
      boost::mpi::communicator &world,
      int max_subtree_size_in,
      DistributedTableType *query_table_in,
      DistributedTableType *reference_table_in,
      TaskQueueType *task_queue_in) {

      // The maximum number of points to hold at a given moment.
      remaining_extra_points_to_hold_ =
        max_subtree_size_in * world.size();

      // Set the pointer to the task queue.
      task_queue_ = task_queue_in;

      // Initialize the stage.
      stage_ = 0;

      // The maximum number of neighbors.
      max_stage_ = static_cast<unsigned int>(log2(world.size()));

      // Set the local table.
      local_table_ = reference_table_in->local_table();

      // Preallocate the cache.
      message_cache_.resize(world.size());
      message_send_request_.resize(world.size());

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
      SubTableRouteRequestType > &hashed_essential_reference_subtrees_to_send) {

      // The ID of the received subtables.
      std::vector< boost::tuple<int, int, int, int> > received_subtable_ids;

      // If the number of processes is only one, then don't bother
      // since there is nothing to exchange.
      if(world.size() == 1) {
        return;
      }

      if(enter_stage_) {

        // Clear the list of received subtables in this round.
        received_subtable_ids.resize(0);

        // At the start of each phase (stage == 0), dequeue something
        // from the hashed list.
        if(stage_ == 0) {

          // The status and the object to be copied onto.
          MessageType &new_self_send_request_object =
            message_cache_[ world.rank()];
          if(hashed_essential_reference_subtrees_to_send.size() > 0) {

            // Examine the back of the route request list.
            SubTableRouteRequestType &route_request =
              hashed_essential_reference_subtrees_to_send.back();

            // Prepare the initial subtable to send.
            new_self_send_request_object.subtable_route().Init(
              world, route_request);
            new_self_send_request_object.subtable_route().set_object_is_valid_flag(true);

            // Pop it from the route request list.
            hashed_essential_reference_subtrees_to_send.pop_back();
          }
          else {

            // Prepare an empty message.
            new_self_send_request_object.subtable_route().Init(world);
            new_self_send_request_object.subtable_route().add_destinations(world);
          }
          if(queued_up_completed_computation_.size() > 0) {

            // Examine the back of the route request list.
            EnergyRouteRequestType &route_request =
              queued_up_completed_computation_.back();

            // Prepare the initial subtable to send.
            new_self_send_request_object.energy_route().Init(
              world, route_request);
            new_self_send_request_object.energy_route().set_object_is_valid_flag(true);

            // Pop it from the route request list.
            queued_up_completed_computation_.pop_back();
          }
          else {

            // Prepare an empty message for the energy portion.
            new_self_send_request_object.energy_route().Init(world);
            new_self_send_request_object.energy_route().add_destinations(world);
            new_self_send_request_object.energy_route().object() = 0;
          }

          // Set the originating rank of the message.
          new_self_send_request_object.set_originating_rank(world.rank());
          new_self_send_request_object.energy_route().set_object_is_valid_flag(true);
        }

        // Exchange with the neighbors.
        unsigned int num_subtables_to_exchange = (1 << stage_);
        unsigned int neighbor = world.rank() ^(1 << stage_);
        unsigned int lower_bound_send = (world.rank() >> stage_) << stage_;
        for(unsigned int i = 0; i < num_subtables_to_exchange; i++) {
          unsigned int subtable_send_index = i + lower_bound_send;
          MessageType &send_request_object =
            message_cache_[ subtable_send_index ];
          send_request_object.next_destination(world);

          // For each subtable sent, we expect something from the neighbor.
          message_send_request_[i] =
            world.isend(
              neighbor, core::parallel::MessageTag::ROUTE_SUBTABLE,
              send_request_object);
        }

        // Receive from the neighbor.
        unsigned int num_subtables_received = 0;
        while(num_subtables_received < num_subtables_to_exchange) {

          if(boost::optional< boost::mpi::status > l_status =
                world.iprobe(
                  neighbor,
                  core::parallel::MessageTag::ROUTE_SUBTABLE)) {

            // Receive the subtable.
            MessageType tmp_route_request;
            tmp_route_request.subtable_route().object().Init(neighbor, false);
            world.recv(
              neighbor,
              core::parallel::MessageTag::ROUTE_SUBTABLE,
              tmp_route_request);
            int cache_id =
              tmp_route_request.originating_rank();
            tmp_route_request.subtable_route().object().set_cache_block_id(cache_id);

            // If this subtable is needed by the calling process, then
            // update the list of subtables received.
            num_subtables_received++;

            message_cache_[ cache_id ] = tmp_route_request;
            MessageType &route_request = message_cache_[cache_id];

            // If the received subtable is valid,
            if(route_request.subtable_route().object_is_valid()) {

              // Lock the subtable equal to the number of remaining
              // phases.
              this->LockCache(cache_id, max_stage_ - stage_ - 1);

              // If the subtable is needed by the process, then add
              // it to its task list.
              if(route_request.subtable_route().remove_from_destination_list(world.rank())) {
                received_subtable_ids.push_back(
                  boost::make_tuple(
                    route_request.subtable_route().object().table()->rank(),
                    route_request.subtable_route().object().start_node()->begin(),
                    route_request.subtable_route().object().start_node()->count(),
                    cache_id));
              }
            }
            else {
              this->ClearSubTable_(cache_id);
            }

            // Update the energy count.
            if(route_request.energy_route().remove_from_destination_list(world.rank()) &&
                route_request.energy_route().object_is_valid()) {
              task_queue_->decrement_remaining_global_computation(
                route_request.energy_route().object());
            }
          }
        }

        // Wait until all sends are done.
        boost::mpi::wait_all(
          message_send_request_.begin(),
          message_send_request_.begin() + num_subtables_to_exchange);

        // For every valid send, unlock its cache.
        for(unsigned int i = 0; i < num_subtables_to_exchange; i++) {
          unsigned int process_rank = i + lower_bound_send;
          if(process_rank != static_cast<unsigned int>(world.rank()) &&
              message_cache_[process_rank].subtable_route().object_is_valid()) {
            this->ReleaseCache(world, process_rank, 1);
          }
        }

        // Generate more tasks.
        task_queue_->GenerateTasks(world, metric_in, received_subtable_ids);

        // Initiate load balancing with the neighbor.
        // LoadBalance_(world, neighbor);

        // Increment the stage when done, and turn off the stage flag.
        stage_ = (stage_ + 1) % max_stage_;
        enter_stage_ = false;

      } // end of the case entering the stage.

      if(! enter_stage_) {

        // Test whether the stage can be entered.
        this->ReadyForStage_(world);
      }
    }
};
}
}

#endif
