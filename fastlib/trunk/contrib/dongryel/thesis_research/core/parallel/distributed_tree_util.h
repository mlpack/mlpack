/** @file distributed_tree_util.h
 *
 *  The common utility functions used for building a distributed
 *  binary tree using a vanilla approach.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_TREE_UTIL_H
#define CORE_PARALLEL_DISTRIBUTED_TREE_UTIL_H

#include <numeric>
#include <boost/mpi.hpp>
#include "core/table/sample_dense_matrix.h"

namespace core {
namespace parallel {

class DistributedTreeExtraUtil {

  public:
    /** @brief Returns the left and the right destinations when
     *         shuffling the data.
     */
    static void left_and_right_destinations(
      boost::mpi::communicator &comm, int *left_rank, int *right_rank,
      bool *color) {
      int threshold = comm.size() / 2;
      if(left_rank != NULL) {
        *left_rank =
          (comm.rank() < threshold) ? comm.rank() : comm.rank() - threshold;
      }
      if(right_rank != NULL) {
        *right_rank = (comm.rank() < threshold) ?
                      comm.rank() + threshold : comm.rank();
      }
      if(color != NULL) {
        *color = (comm.rank() < threshold);
      }
    }
};

template<typename DistributedTableType>
class DistributedTreeUtil {

  public:

    /** @brief The tree spec.
     */
    typedef typename DistributedTableType::TableType::TreeSpecType TreeSpecType;

    /** @brief The table type.
     */
    typedef typename DistributedTableType::TableType TableType;

    /** @brief The tree type.
     */
    typedef typename DistributedTableType::TreeType TreeType;

    /** @brief The bound type.
     */
    typedef typename TreeType::BoundType BoundType;

    /** @brief The old from new index type.
     */
    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

    /** @brief The sample matrix type for reshuffling.
     */
    typedef core::table::SampleDenseMatrix<OldFromNewIndexType>
    SampleDenseMatrixType;

  public:

    static void SetupGatherPointers(
      TableType &sampled_table, const std::vector<int> &counts,
      std::vector< SampleDenseMatrixType > *gather_pointers_out) {

      // Have the pointers point to the right position based on the
      // prefix sum position.
      gather_pointers_out->resize(counts.size());
      int starting_column_index = 0;
      for(unsigned int i = 0; i < counts.size(); i++) {
        (*gather_pointers_out)[i].Init(
          sampled_table.data(), sampled_table.old_from_new(),
          starting_column_index, counts[i]);
        starting_column_index += counts[i];
      }
    }

    /** @brief Reshuffle points across each process.
     *
     *  @param world The communicator.
     *
     *  @param assigned_point_indices The list of assigned point
     *         indices. The $i$-th position denotes the list of points
     *         that should be transferred to the $i$-th process.
     *
     *  @param membership_counts_per_node The list of sizes of
     *         assigned point indices. The $i$-th position denotes the
     *         number of points assigned to the $i$-th process
     *         originating from the current process.
     */
    static void ReshufflePoints(
      boost::mpi::communicator &world,
      const std::vector< std::vector<int> > &assigned_point_indices,
      const std::vector<int> &membership_counts_per_node,
      DistributedTableType *distributed_table_in,
      int n_attributes_in) {

      // Do an all-to-all to figure out the new table sizes for each
      // process.
      std::vector<int> reshuffled_contributions;
      boost::mpi::all_to_all(
        world, membership_counts_per_node, reshuffled_contributions);
      int total_num_points_owned =
        std::accumulate(
          reshuffled_contributions.begin(), reshuffled_contributions.end(), 0);

      // Allocate a new table to get all the points.
      TableType *new_local_table =
        (core::table::global_m_file_) ?
        core::table::global_m_file_->Construct<TableType>() : new TableType();
      new_local_table->Init(
        n_attributes_in, total_num_points_owned,
        world.rank());

      // Setup points so that they will be reshuffled.
      std::vector<SampleDenseMatrixType> points_to_be_distributed;
      points_to_be_distributed.resize(world.size());
      for(unsigned int i = 0; i < points_to_be_distributed.size(); i++) {
        points_to_be_distributed[i].Init(
          distributed_table_in->local_table()->data(),
          distributed_table_in->local_table()->old_from_new(),
          assigned_point_indices[i]);
      }
      std::vector<SampleDenseMatrixType> reshuffled_points;
      SetupGatherPointers(
        *new_local_table, reshuffled_contributions, &reshuffled_points);

      // Important that each process takes care of the exporting of
      // its own part.
      points_to_be_distributed[world.rank()].Export(
        reshuffled_points[world.rank()].matrix(),
        reshuffled_points[world.rank()].old_from_new(),
        reshuffled_points[world.rank()].starting_column_index());
      boost::mpi::all_to_all(
        world, points_to_be_distributed, reshuffled_points);

      // Set it to the newly shuffled table.
      distributed_table_in->set_local_table(new_local_table);
    }
};
}
}

#endif
