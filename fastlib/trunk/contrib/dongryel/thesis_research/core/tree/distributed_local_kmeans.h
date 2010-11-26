/** @file distributed_local_kmeans.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_DISTRIBUTED_LOCAL_KMEANS_H
#define CORE_TREE_DISTRIBUTED_LOCAL_KMEANS_H

#include <boost/mpi.hpp>
#include "core/table/dense_point.h"

namespace core {
namespace tree {
class DistributedLocalKMeans {
  public:

    void Compute(
      boost::mpi::communicator &comm,
      const core::table::DensePoint &starting_centroid,
      int neighbor_radius, int num_outer_loop_iterations) {

      // Every process collects the local centers from the process ID
      // within the specified neighbor_radius.
      core::table::DensePoint local_centroid;
      local_centroid.Copy(starting_centroid);

      // The list of centroids left of the current process.
      std::vector< core::table::DensePoint > left_centroids;
      left_centroids.resize(std::min(comm.rank(), neighbor_radius));
      std::vector< boost::mpi::request > left_send_requests;
      left_send_requests.resize(left_centroids.size());
      std::vector< boost::mpi::request > left_receive_requests;
      left_receive_requests.resize(left_centroids.size());

      // The list of centroids right of the current process.
      std::vector< core::table::DensePoint > right_centroids;
      right_centroids.resize(std::min(
                               comm.size() - comm.rank() - 1, neighbor_radius));
      std::vector< boost::mpi::request > right_send_requests;
      right_send_requests.resize(right_centroids.size());
      std::vector< boost::mpi::request > right_receive_requests;
      right_receive_requests.resize(right_centroids.size());

      for(int outer_loop = 0; outer_loop < num_outer_loop_iterations;
          outer_loop++) {

        for(unsigned int i = 1; i <= left_centroids.size(); i++) {

          // Send and receive.
          left_send_requests[i] =
            comm.isend(comm.rank() - i, i, local_centroid);
          left_receive_requests[i] =
            comm.irecv(
              comm.rank() - i, neighbor_radius + i, left_centroids[i - 1]);
        }
        for(unsigned int i = 1; i <= right_centroids.size(); i++) {

          // Send and receive.
          right_send_requests[i] =
            comm.isend(
              comm.rank() + i, neighbor_radius + i, local_centroid);
          right_receive_requests[i] =
            comm.irecv(comm.rank() + i, i, right_centroids[i - 1]);
        }

        // Wait for all send/receive requests to be completed.
        boost::mpi::wait_all(
          left_send_requests.begin(), left_send_requests.end());
        boost::mpi::wait_all(
          left_receive_requests.begin(), left_receive_requests.end());
        boost::mpi::wait_all(
          right_send_requests.begin(), right_send_requests.end());
        boost::mpi::wait_all(
          right_receive_requests.begin(), right_receive_requests.end());


      }
    }
};
};
};

#endif
