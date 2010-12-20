/** @file distributed_local_kmeans.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_DISTRIBUTED_LOCAL_KMEANS_H
#define CORE_TREE_DISTRIBUTED_LOCAL_KMEANS_H

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include "core/math/linear_algebra.h"
#include "core/table/dense_point.h"

namespace core {
namespace tree {
class DistributedLocalKMeans {
  private:

    class CentroidInfo {
      private:

        friend class boost::serialization::access;

      private:

        core::table::DensePoint centroid_;

        int num_points_;

      public:

        void Init(int length_in) {
          centroid_.Init(length_in);
        }

        const core::table::DensePoint &centroid() const {
          return centroid_;
        }

        core::table::DensePoint &centroid() {
          return centroid_;
        }

        int num_points() const {
          return num_points_;
        }

        CentroidInfo() {
          num_points_ = 0;
        }

        void Add(const CentroidInfo &centroid_in) {
          double factor =
            static_cast<double>(num_points_) /
            static_cast<double>(num_points_ + centroid_in.num_points());
          core::math::Scale(factor, &centroid_);
          core::math::AddExpert(
            1.0 - factor, centroid_in.centroid(), &centroid_);
          num_points_ = num_points_ + centroid_in.num_points();
        }

        void Add(const core::table::DensePoint &point_in) {
          double factor = static_cast<double>(num_points_) /
                          static_cast<double>(num_points_ + 1);
          core::math::Scale(factor, &centroid_);
          core::math::AddExpert(1.0 - factor, point_in, &centroid_);
          num_points_++;
        }

        void Reset() {
          centroid_.SetZero();
          num_points_ = 0;
        }

        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) {
          ar & centroid_;
          ar & num_points_;
        }
    };

  private:

    std::vector< CentroidInfo > left_centroids_;

    std::vector< CentroidInfo > right_centroids_;

    std::vector< boost::mpi::request > left_send_requests_;

    std::vector< boost::mpi::request > left_receive_requests_;

    std::vector< boost::mpi::request > right_send_requests_;

    std::vector< boost::mpi::request > right_receive_requests_;

    CentroidInfo local_centroid_;

    std::vector< CentroidInfo > tmp_left_centroids_;

    std::vector< CentroidInfo > tmp_recv_from_left_;

    std::vector< CentroidInfo > tmp_right_centroids_;

    std::vector< CentroidInfo > tmp_recv_from_right_;

  private:

    template<typename TableType>
    void SynchronizeCentroids_(
      boost::mpi::communicator &comm, TableType &local_table_in,
      int neighbor_radius, const std::vector<int> &point_assignments) {

      // Reset the contribution list.
      local_centroid_.Reset();
      for(unsigned int i = 0; i < tmp_left_centroids_.size(); i++) {
        tmp_left_centroids_[i].Reset();
      }
      for(unsigned int i = 0; i < tmp_right_centroids_.size(); i++) {
        tmp_right_centroids_[i].Reset();
      }

      for(unsigned int i = 0; i < point_assignments.size(); i++) {
        core::table::DensePoint point;
        local_table_in.get(i, &point);

        if(point_assignments[i] == comm.rank()) {
          local_centroid_.Add(point);
        }
        else if(point_assignments[i] < comm.rank()) {
          int destination_index = comm.rank() - point_assignments[i] - 1;
          tmp_left_centroids_[destination_index].Add(point);
        }
        else {
          int destination_index = point_assignments[i] - comm.rank() - 1;
          tmp_right_centroids_[destination_index].Add(point);
        }
      }

      // Wait until the local centroids are updated for all processes.
      comm.barrier();

      // After local contributions are updated, send to the left and
      // to the right neighbors. Also receive from the neighbors.
      for(unsigned int i = 1; i <= left_centroids_.size(); i++) {
        left_send_requests_[i - 1] =
          comm.isend(comm.rank() - i, i, tmp_left_centroids_[i - 1]);
        left_receive_requests_[i - 1] =
          comm.irecv(
            comm.rank() - i, neighbor_radius + i,
            tmp_recv_from_left_[i - 1]);
      }
      for(unsigned int i = 1; i <= right_centroids_.size(); i++) {

        // Send and receive.
        right_send_requests_[i - 1] =
          comm.isend(
            comm.rank() + i, neighbor_radius + i,
            tmp_right_centroids_[i - 1]);
        right_receive_requests_[i - 1] =
          comm.irecv(
            comm.rank() + i, i, tmp_recv_from_right_[i - 1]);
      }

      // Wait for all send/receive requests to be completed.
      boost::mpi::wait_all(
        left_send_requests_.begin(), left_send_requests_.end());
      boost::mpi::wait_all(
        left_receive_requests_.begin(), left_receive_requests_.end());
      boost::mpi::wait_all(
        right_send_requests_.begin(), right_send_requests_.end());
      boost::mpi::wait_all(
        right_receive_requests_.begin(), right_receive_requests_.end());

      // Update the local centroid information.
      for(unsigned int i = 0; i < tmp_recv_from_left_.size(); i++) {
        local_centroid_.Add(tmp_recv_from_left_[i]);
      }
      for(unsigned int i = 0; i < tmp_recv_from_right_.size(); i++) {
        local_centroid_.Add(tmp_recv_from_right_[i]);
      }
    }

  public:

    template<typename TableType>
    void Compute(
      boost::mpi::communicator &comm,
      const core::metric_kernels::AbstractMetric &metric,
      TableType &local_table_in,
      int neighbor_radius, int num_outer_loop_iterations,
      core::table::DensePoint &starting_centroid,
      int *total_num_points_owned,
      std::vector<int> *point_assignments_out) {

      // Every process collects the local centers from the process ID
      // within the specified neighbor_radius.
      local_centroid_.centroid().Copy(starting_centroid);

      // The list of centroids left of the current process.
      left_centroids_.resize(std::min(comm.rank(), neighbor_radius));
      left_send_requests_.resize(left_centroids_.size());
      left_receive_requests_.resize(left_centroids_.size());
      tmp_left_centroids_.resize(std::min(comm.rank(), neighbor_radius));
      tmp_recv_from_left_.resize(std::min(comm.rank(), neighbor_radius));
      for(unsigned int i = 0; i < tmp_left_centroids_.size(); i++) {
        tmp_left_centroids_[i].Init(starting_centroid.length());
      }

      // The list of centroids right of the current process.
      right_centroids_.resize(
        std::min(
          comm.size() - comm.rank() - 1, neighbor_radius));
      right_send_requests_.resize(right_centroids_.size());
      right_receive_requests_.resize(right_centroids_.size());
      tmp_right_centroids_.resize(
        std::min(
          comm.size() - comm.rank() - 1, neighbor_radius));
      tmp_recv_from_right_.resize(
        std::min(
          comm.size() - comm.rank() - 1, neighbor_radius));
      for(unsigned int i = 0; i < tmp_right_centroids_.size(); i++) {
        tmp_right_centroids_[i].Init(starting_centroid.length());
      }

      // List of assignments for each point in the table.
      point_assignments_out->resize(local_table_in.n_entries());

      // The outer loop.
      for(
        int outer_loop = 0; outer_loop < num_outer_loop_iterations;
        outer_loop++) {

        for(unsigned int i = 1; i <= left_centroids_.size(); i++) {

          // Send and receive.
          left_send_requests_[i - 1] =
            comm.isend(comm.rank() - i, i, local_centroid_);
          left_receive_requests_[i - 1] =
            comm.irecv(
              comm.rank() - i, neighbor_radius + i, left_centroids_[i - 1]);
        }
        for(unsigned int i = 1; i <= right_centroids_.size(); i++) {

          // Send and receive.
          right_send_requests_[i - 1] =
            comm.isend(
              comm.rank() + i, neighbor_radius + i, local_centroid_);
          right_receive_requests_[i - 1] =
            comm.irecv(comm.rank() + i, i, right_centroids_[i - 1]);
        }

        // Wait for all send/receive requests to be completed.
        boost::mpi::wait_all(
          left_send_requests_.begin(), left_send_requests_.end());
        boost::mpi::wait_all(
          left_receive_requests_.begin(), left_receive_requests_.end());
        boost::mpi::wait_all(
          right_send_requests_.begin(), right_send_requests_.end());
        boost::mpi::wait_all(
          right_receive_requests_.begin(), right_receive_requests_.end());

        // Synchronize.
        comm.barrier();

        // For each point in the table,
        for(int p = 0; p < local_table_in.n_entries(); p++) {
          core::table::DensePoint point;
          local_table_in.get(p, &point);

          // Compute the closest center.
          double min_squared_distance = metric.DistanceSq(
                                          point, local_centroid_.centroid());
          int min_index = comm.rank();
          for(unsigned int lc = 0; lc < left_centroids_.size(); lc++) {
            double squared_distance =
              metric.DistanceSq(
                point, left_centroids_[lc].centroid());
            if(squared_distance < min_squared_distance) {
              min_squared_distance = squared_distance;
              min_index = comm.rank() - lc - 1;
            }
          }
          for(unsigned int rc = 0; rc < right_centroids_.size(); rc++) {
            double squared_distance =
              metric.DistanceSq(
                point, right_centroids_[rc].centroid());
            if(squared_distance < min_squared_distance) {
              min_squared_distance = squared_distance;
              min_index = comm.rank() + rc + 1;
            }
          }

          // Assign the point.
          (*point_assignments_out)[p] = min_index;

        } // end of iterating over each point.

        // Recompute the local centroid contribution and send to the
        // left and to the right.
        SynchronizeCentroids_(
          comm, local_table_in, neighbor_radius, *point_assignments_out);

        // Synchronize before continuing the outer iteration.
        comm.barrier();

      } // end of outer iterations.

      // Copy the final ending position and the total number of points
      // owned by this centroid.
      starting_centroid.CopyValues(local_centroid_.centroid());
      *total_num_points_owned = local_centroid_.num_points();
    }
};
};
};

#endif
