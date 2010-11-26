/** @file distributed_local_kmeans.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_DISTRIBUTED_LOCAL_KMEANS_H
#define CORE_TREE_DISTRIBUTED_LOCAL_KMEANS_H

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include "core/table/dense_point.h"

namespace core {
namespace tree {
class DistributedLocalKMeans {
  private:

    class CentroidInfo {
      private:

        friend class boost::serialization::access;

      public:

        core::table::DensePoint centroid_;

        int num_points_;

      public:

        CentroidInfo() {
          num_points_ = 0;
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

    std::vector< core::table::DensePoint > left_centroids_;

    std::vector< core::table::DensePoint > right_centroids_;

    std::vector< boost::mpi::request > left_send_requests_;

    std::vector< boost::mpi::request > left_receive_requests_;

    std::vector< boost::mpi::request > right_send_requests_;

    std::vector< boost::mpi::request > right_receive_requests_;

    CentroidInfo local_centroid_;

    std::vector< CentroidInfo > tmp_left_centroids_;

    std::vector< CentroidInfo > tmp_right_centroids_;

    std::vector<int> point_assignments_;

  public:

    template<typename TableType>
    void Compute(
      boost::mpi::communicator &comm,
      const core::metric_kernels::AbstractMetric &metric,
      TableType &local_table_in,
      const core::table::DensePoint &starting_centroid,
      int neighbor_radius, int num_outer_loop_iterations) {

      // Every process collects the local centers from the process ID
      // within the specified neighbor_radius.
      local_centroid_.centroid_.Copy(starting_centroid);

      // The list of centroids left of the current process.
      left_centroids_.resize(std::min(comm.rank(), neighbor_radius));
      left_send_requests_.resize(left_centroids_.size());
      left_receive_requests_.resize(left_centroids_.size());
      tmp_left_centroids_.resize(std::min(comm.rank(), neighbor_radius));

      // The list of centroids right of the current process.
      right_centroids_.resize(
        std::min(
          comm.size() - comm.rank() - 1, neighbor_radius));
      right_send_requests_.resize(right_centroids_.size());
      right_receive_requests_.resize(right_centroids_.size());
      tmp_right_centroids_.resize(
        std::min(
          comm.size() - comm.rank() - 1, neighbor_radius));

      // List of assignments for each point in the table.
      point_assignments_.resize(local_table_in.n_entries());

      // The outer loop.
      for(int outer_loop = 0; outer_loop < num_outer_loop_iterations;
          outer_loop++) {

        for(unsigned int i = 1; i <= left_centroids_.size(); i++) {

          // Send and receive.
          left_send_requests_[i] =
            comm.isend(comm.rank() - i, i, local_centroid_.centroid_);
          left_receive_requests_[i] =
            comm.irecv(
              comm.rank() - i, neighbor_radius + i, left_centroids_[i - 1]);
        }
        for(unsigned int i = 1; i <= right_centroids_.size(); i++) {

          // Send and receive.
          right_send_requests_[i] =
            comm.isend(
              comm.rank() + i, neighbor_radius + i, local_centroid_.centroid_);
          right_receive_requests_[i] =
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
                                          point, local_centroid_.centroid_);
          double min_index = comm.rank();
          for(unsigned int lc = 0; lc < left_centroids_.size(); lc++) {
            double squared_distance = metric.DistanceSq(
                                        point, left_centroids_[lc]);
            if(squared_distance < min_squared_distance) {
              min_squared_distance = squared_distance;
              min_index = comm.rank() - lc - 1;
            }
          }
          for(unsigned int rc = 0; rc < right_centroids_.size(); rc++) {
            double squared_distance = metric.DistanceSq(
                                        point, right_centroids_[rc]);
            if(squared_distance < min_squared_distance) {
              min_squared_distance = squared_distance;
              min_index = comm.rank() - rc + 1;
            }
          }

          // Assign the point.
          point_assignments_[p] = min_index;

        } // end of iterating over each point.

        // Recompute the local centroid contribution and send to the
        // left and to the right.

        // Synchronize before continuing the outer iteration.
        comm.barrier();

      } // end of outer iterations.
    }
};
};
};

#endif
