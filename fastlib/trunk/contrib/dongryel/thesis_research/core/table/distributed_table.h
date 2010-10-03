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
    }

    PointRequestMessage(int source_rank_in, int point_id_in) {
      source_rank_ = source_rank_in;
      point_id_ = point_id_in;
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

class DistributedTable: public boost::noncopyable {

    typedef core::tree::GeneralBinarySpaceTree < core::tree::BallBound <
    core::table::DensePoint > > TreeType;

  private:

    int rank_;

    core::table::Table *owned_table_;

    std::vector<int> local_n_entries_;

    TreeType *global_tree_;

    boost::mpi::communicator *comm_;

    boost::shared_ptr<boost::thread> table_thread_;

  public:

    int rank() const {
      return rank_;
    }

    bool IsIndexed() const {
      return global_tree_ != NULL;
    }

    DistributedTable() {
      rank_ = -1;
      owned_table_ = NULL;
      global_tree_ = NULL;
    }

    ~DistributedTable() {
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
      table_thread_->detach();
    }

    void Save(const std::string &file_name) const {

    }

    void IndexData(
      const core::metric_kernels::AbstractMetric &metric_in, int leaf_size) {

    }

    void server() const {

      std::vector<double> point_vector(this->n_attributes(), 0);
      while(true) {

        // Probe the message queue.
        do {
          if(comm_->iprobe(
                boost::mpi::any_source,
                core::table::DistributedTableMessage::REQUEST_POINT)) {
            break;
          }
          else {
          }
        }
        while(true);

        // Try to receive the message.
        core::table::PointRequestMessage point_request_message;
        boost::mpi::request receive_request =
          comm_->irecv(
            boost::mpi::any_source,
            core::table::DistributedTableMessage::REQUEST_POINT,
            point_request_message);
        receive_request.wait();

        // Copy the point out.
        owned_table_->get(point_request_message.point_id(), &point_vector);

        // Send back the point to the requester.
        boost::mpi::request send_request =
          comm_->isend(
            point_request_message.source_rank(),
            core::table::DistributedTableMessage::RECEIVE_POINT,
            point_vector);
        send_request.wait();
      }
    }

    void get(
      int requested_rank, int point_id,
      core::table::DensePoint *entry) const {

      // If owned by the process, just return the point. Otherwise, we
      // need to send an MPI request to the process holding the
      // required resource.
      if(rank_ == requested_rank) {
        owned_table_->get(point_id, entry);
      }
      else {

        // The point request message.
        core::table::PointRequestMessage point_request_message(rank_, point_id);

        // We receive the point in the form of std::vector.
        std::vector<double> received_point_vector;

        // Inform the source processor that this processor needs data!
        boost::mpi::request point_request =
          comm_->isend(
            requested_rank,
            core::table::DistributedTableMessage::REQUEST_POINT,
            point_request_message);
        point_request.wait();

        // Wait for the source processor's answer.
        boost::mpi::request point_receive_request =
          comm_->irecv(
            requested_rank,
            core::table::DistributedTableMessage::RECEIVE_POINT,
            received_point_vector);
        point_receive_request.wait();

        // Copy the data.
        entry->Init(received_point_vector);
      }
    }

    void PrintTree() const {
      global_tree_->Print();
    }
};
};
};

#endif
