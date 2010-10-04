/** @file distributed_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_TABLE_H
#define CORE_TABLE_DISTRIBUTED_TABLE_H

#include <armadillo>
#include "boost/mpi.hpp"
#include "boost/mpi/collectives.hpp"
#include "boost/thread.hpp"
#include "boost/serialization/string.hpp"
#include "core/table/table.h"

namespace core {
namespace table {

class PointRequestMessage {
  private:
    int source_rank_;

    int point_id_;

    friend class boost::serialization::access;

  public:

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & source_rank_;
      ar & point_id_;
    }

    PointRequestMessage() {
      Reset();
    }

    PointRequestMessage(int source_rank_in, int point_id_in) {
      source_rank_ = source_rank_in;
      point_id_ = point_id_in;
    }

    bool is_valid() const {
      return source_rank_ >= 0 && point_id_ >= 0;
    }

    void Reset() {
      source_rank_ = -1;
      point_id_ = -1;
    }

    int source_rank() const {
      return source_rank_;
    }

    int point_id() const {
      return point_id_;
    }
};

class DistributedTableMessage {
  public:
    enum DistributedTableRequest { REQUEST_POINT, RECEIVE_POINT };
};

class Mailbox {
  public:

    static const int incoming_request_mailbox_size = 10;

    boost::mutex mutex_;

    boost::condition_variable point_ready_cond_;

    std::pair <
    boost::mpi::request,
          core::table::PointRequestMessage > *incoming_request_mailbox_;

    std::vector<int> free_slots_;

    boost::mpi::request outgoing_request_;

    boost::mpi::request incoming_receive_request_;

    int incoming_receive_source_;

    std::vector<double> incoming_point_;

    std::vector<double> outgoing_point_;

  public:

    bool is_active() const {
      return false;
    }

    Mailbox() {
      incoming_request_mailbox_ =
        new std::pair <
      boost::mpi::request, core::table::PointRequestMessage > [
        incoming_request_mailbox_size];
      free_slots_.resize(incoming_request_mailbox_size);
      for(unsigned int i = 0; i < free_slots_.size(); i++) {
        free_slots_[i] = i;
      }
    }

    ~Mailbox() {
      delete[] incoming_request_mailbox_;
      incoming_request_mailbox_ = NULL;
    }

    bool incoming_request_mailbox_is_full() const {
      return free_slots_.size() == 0;
    }
};

class DistributedTable: public boost::noncopyable {

    typedef core::tree::GeneralBinarySpaceTree < core::tree::BallBound <
    core::table::DensePoint > > TreeType;

  private:

    bool destruct_flag_;

    int rank_;

    core::table::Table *owned_table_;

    std::vector<int> local_n_entries_;

    TreeType *global_tree_;

    boost::mpi::communicator *comm_;

    boost::shared_ptr<boost::thread> table_thread_;

    core::table::Mailbox mailbox_;

  public:

    int rank() const {
      return rank_;
    }

    bool IsIndexed() const {
      return global_tree_ != NULL;
    }

    DistributedTable() {
      destruct_flag_ = false;
      rank_ = -1;
      owned_table_ = NULL;
      global_tree_ = NULL;
    }

    ~DistributedTable() {
      comm_->barrier();

      // Set the flag so that the server thread can access it and kill
      // itself.
      destruct_flag_ = true;

      if(owned_table_ != NULL) {
        delete owned_table_;
        rank_ = -1;
        owned_table_ = NULL;
      }
      if(global_tree_ != NULL) {
        delete global_tree_;
        global_tree_ = NULL;
      }
    }

    const TreeType::BoundType &get_node_bound(TreeType *node) const {
      return node->bound();
    }

    TreeType::BoundType &get_node_bound(TreeType *node) {
      return node->bound();
    }

    TreeType *get_node_left_child(TreeType *node) {
      return node->left();
    }

    TreeType *get_node_right_child(TreeType *node) {
      return node->right();
    }

    bool node_is_leaf(TreeType *node) const {
      return node->is_leaf();
    }

    core::tree::AbstractStatistic *&get_node_stat(TreeType *node) {
      return node->stat();
    }

    int get_node_count(TreeType *node) const {
      return node->count();
    }

    TreeType *get_tree() {
      return global_tree_;
    }

    int n_attributes() const {
      return owned_table_->n_attributes();
    }

    int local_n_entries(int rank_in) const {
      return local_n_entries_[rank_in];
    }

    int local_n_entries() const {
      return owned_table_->n_entries();
    }

    void Init(
      int rank_in,
      const std::string &file_name,
      boost::mpi::communicator *communicator_in) {

      rank_ = rank_in;
      comm_ = communicator_in;
      owned_table_ = new core::table::Table();
      owned_table_->Init(file_name);

      // Allocate the vector for storing the number of entries for all
      // the tables in the world, and do an all-gather operation to
      // find out all the sizes.
      boost::mpi::all_gather(
        *comm_, owned_table_->n_entries(), local_n_entries_);

      // Start the server for giving out points.
      table_thread_ = boost::shared_ptr<boost::thread>(
                        new boost::thread(
                          boost::bind(
                            &core::table::DistributedTable::server,
                            this)));

      comm_->barrier();
      table_thread_->detach();
    }

    void Save(const std::string &file_name) const {

    }

    void IndexData(
      const core::metric_kernels::AbstractMetric &metric_in, int leaf_size) {

    }

    void server() {

      while(true) {

        // Probe the message queue for the point request, and do an
        // asynchronous receive first and buffer the receive.
        while(mailbox_.incoming_request_mailbox_is_full() == false &&
              comm_->iprobe(
                boost::mpi::any_source,
                core::table::DistributedTableMessage::REQUEST_POINT)) {

          int free_slot = mailbox_.free_slots_[
                            mailbox_.free_slots_.size() - 1];
          mailbox_.incoming_request_mailbox_[ free_slot ].first =
            comm_->irecv(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::REQUEST_POINT,
              mailbox_.incoming_request_mailbox_[free_slot].second);

          // Decrement the mail box free slot.
          mailbox_.free_slots_.resize(mailbox_.free_slots_.size() - 1);
        }

        // See if the current outgoing request is done.
        if(mailbox_.outgoing_request_.test()) {

          // Try to see if any of the send requests know where to send
          // stuffs to, and start the asynchronous transfer of the
          // appropriate point to the requestor. Right now, the buffer
          // size is 1 point, so I cannot send more than one point
          // without increasing the buffer size.
          for(int i = 0;
              i < core::table::Mailbox::incoming_request_mailbox_size; i++) {

            std::pair< boost::mpi::request, core::table::PointRequestMessage >
            &incoming_request = mailbox_.incoming_request_mailbox_[i];

            if(incoming_request.second.is_valid() &&
                incoming_request.first.test()) {

              // Get the reference to the incoming request to be
              // fulfilled.
              core::table::PointRequestMessage &to_be_flushed =
                mailbox_.incoming_request_mailbox_[i].second;

              // Copy the point out.
              owned_table_->get(
                to_be_flushed.point_id(),
                &mailbox_.outgoing_point_);

              // Send back the point to the requester.
              mailbox_.outgoing_request_ =
                comm_->isend(
                  to_be_flushed.source_rank(),
                  core::table::DistributedTableMessage::RECEIVE_POINT,
                  mailbox_.outgoing_point_);

              // This mail slot is free.
              to_be_flushed.Reset();
              mailbox_.free_slots_.push_back(i);

              break;
            }
          }
        }

        // Check if any of the requested point from this processor is
        // ready to be received.
        if(comm_->iprobe(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::RECEIVE_POINT)) {
          mailbox_.incoming_receive_request_ =
            comm_->irecv(mailbox_.incoming_receive_source_,
                         core::table::DistributedTableMessage::RECEIVE_POINT,
                         mailbox_.incoming_point_);
        }

        // Check whether the incoming receive is done, if so, wake up
        // the sleeping thread.
        if(mailbox_.incoming_receive_request_.test()) {
          mailbox_.point_ready_cond_.notify_one();
        }

        // If the main thread has given a termination signal, and we
        // are done with everything, then the server thread exits.
        if(destruct_flag_ && mailbox_.is_active() == false) {
          break;
        }

      } // end of the loop for handling buffered messages.
    }

    void get(
      int requested_rank, int point_id,
      core::table::DensePoint *entry) {

      // If owned by the process, just return the point. Otherwise, we
      // need to send an MPI request to the process holding the
      // required resource.
      if(rank_ == requested_rank) {
        owned_table_->get(point_id, entry);
      }
      else {

        // The point request message.
        core::table::PointRequestMessage point_request_message(rank_, point_id);

        // Inform the source processor that this processor needs data!
        mailbox_.incoming_receive_source_ = requested_rank;
        comm_->isend(
          requested_rank,
          core::table::DistributedTableMessage::REQUEST_POINT,
          point_request_message);

        // Do a conditional wait until the point is ready.
        boost::unique_lock<boost::mutex> lock(mailbox_.mutex_);
        mailbox_.point_ready_cond_.wait(lock);

        // If we are here, then the point is ready. Copy the point.
        entry->Init(mailbox_.incoming_point_);
      }
    }

    void PrintTree() const {
      global_tree_->Print();
    }
};
};
};

#endif
