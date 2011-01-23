/** @file vanilla_distributed_tree_builder.h
 *
 *  The generic template for building a distributed binary tree using
 *  a vanilla approach.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_VANILLA_DISTRIBUTED_TREE_BUILDER_H
#define CORE_PARALLEL_VANILLA_DISTRIBUTED_TREE_BUILDER_H

#include <algorithm>
#include <numeric>
#include <boost/bind.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include "core/table/memory_mapped_file.h"
#include "core/table/sample_dense_matrix.h"
#include "core/tree/hrect_bound.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace core {
namespace parallel {
template<typename DistributedTableType>
class VanillaDistributedTreeBuilder {

  public:

    /** @brief The tree spec.
     */
    typedef typename DistributedTableType::TableType TreeSpecType;

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

  private:

    /** @brief The pointer to the distributed table. The data will be
     *         reshuffled and exchanged among the MPI processes.
     */
    DistributedTableType *distributed_table_;

    /** @brief The dimensionality.
     */
    int n_attributes_;

  private:

    /** @brief Given the global order in the sorted list, find out
     *         which bin it falls into for a given cumulative
     *         distribution.
     *
     *  @param global_order The global order in the sorted list.
     *  @param start_index The index in the cumulative distribution to
     *  start looking for.
     *  @param desired_cumulative_distribution The cumulative distribution.
     *
     *  @return The bin number the global order belongs to.
     */
    int Locate_(
      int global_order,
      int start_index,
      const std::vector<int> &desired_cumulative_distribution) const {

      for(; start_index <
          static_cast<int>(desired_cumulative_distribution.size()) &&
          global_order >= desired_cumulative_distribution[start_index];
          start_index++);
      return start_index - 1;
    }

    void SetupGatherPointers_(
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
    void ReshufflePoints_(
      boost::mpi::communicator &world,
      const std::vector< std::vector<int> > &assigned_point_indices,
      const std::vector<int> &membership_counts_per_node) {

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
        n_attributes_, total_num_points_owned,
        world.rank());

      // Setup points so that they will be reshuffled.
      std::vector<SampleDenseMatrixType> points_to_be_distributed;
      points_to_be_distributed.resize(world.size());
      for(unsigned int i = 0; i < points_to_be_distributed.size(); i++) {
        points_to_be_distributed[i].Init(
          distributed_table_->local_table()->data(),
          distributed_table_->local_table()->old_from_new(),
          assigned_point_indices[i]);
      }
      std::vector<SampleDenseMatrixType> reshuffled_points;
      SetupGatherPointers_(
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
      distributed_table_->set_local_table(new_local_table);
    }

    /** @brief Do a re-distribution preserving the order such that
     *         each process has roughly the same number of points.
     */
    template<typename MetricType>
    void RedistributeEqually_(
      boost::mpi::communicator &world,
      const MetricType &metric_in,
      const std::vector<int> &sorted_indices_increasing) {

      // Do a scan.
      int cumulative_sum;
      boost::mpi::scan(
        world, distributed_table_->local_table()->n_entries(),
        cumulative_sum, std::plus<int>());
      std::vector<int> cumulative_distribution;
      boost::mpi::all_gather(world, cumulative_sum, cumulative_distribution);

      // Distribute the points roughly equally across all processes.
      int total_num_points = cumulative_distribution.back();
      int points_per_process = total_num_points / world.size();
      int remainder = total_num_points - points_per_process * world.size();
      std::vector<int> desired_cumulative_distribution(world.size(), 0);
      desired_cumulative_distribution[0] =
        (remainder > 0) ? (points_per_process + 1) : points_per_process;
      for(int i = 1; i < world.size(); i++) {
        desired_cumulative_distribution[i] =
          desired_cumulative_distribution[i - 1] + points_per_process;
        if(i < remainder) {
          desired_cumulative_distribution[i]++;
        }
      }
      cumulative_distribution.pop_back();
      desired_cumulative_distribution.pop_back();
      cumulative_distribution.insert(cumulative_distribution.begin(), 0);
      desired_cumulative_distribution.insert(
        desired_cumulative_distribution.begin(), 0);

      // Label each point's destination.
      std::vector< std::vector<int> > assigned_point_indices;
      std::vector<int> membership_counts_per_node(world.size(), 0);
      assigned_point_indices.resize(world.size());
      int previous_destination = 0;
      for(int i = 0; i < distributed_table_->local_table()->n_entries(); i++) {
        int global_order = i + cumulative_distribution[world.rank()];
        int new_destination =
          Locate_(
            global_order, previous_destination,
            desired_cumulative_distribution);
        previous_destination = new_destination;
        assigned_point_indices[new_destination].push_back(i);
        membership_counts_per_node[new_destination]++;
      }
      ReshufflePoints_(
        world, assigned_point_indices, membership_counts_per_node);
    }

    /** @brief Recursively splits a given node creating its children.
     */
    template<typename MetricType>
    static void RecursiveReshuffle_(
      boost::mpi::communicator &comm, const MetricType &metric_in) {

      // If the communicator size is greater than one, then try to split.
      if(comm.size() > 1) {

        // Find the bounding primitive containing all the points
        // belonging to the process belonging to the communicator.
        BoundType bound;
        bound.Init(n_attributes_);
        TreeSpecType::FindBoundFromMatrix(
          comm, metric_in, distributed_table_->table().data(), &bound);

        // Find the split.
        bool can_cut = TreeSpecType::AttemptSplitting(
                         comm, metric_in, distributed_table_);

        if(can_cut) {

          // Split the communicator into two groups here and recurse.
          boost::mpi::communicator left_comm = comm.split(0, comm.rank() % 2);
          boost::mpi::communicator right_comm = comm.split(1, comm.rank() % 2);

          RecursiveReshuffle_(
            left_comm, metric_in, distributed_table);
          RecursiveReshuffle_(
            right_comm, metric_in, distributed_table);
        }
      }
    }

  public:

    /** @brief The default constructor.
     */
    VanillaDistributedTreeBuilder() {
      distributed_table_ = NULL;
      n_attributes_ = 0;
    }

    /** @brief Initialize with a given distributed table with the
     *         given sampling rate for building the top tree.
     *
     *  @param distributed_table_in The distributed table.
     */
    void Init(
      DistributedTableType &distributed_table_in) {

      // This assumes that each distributed table knows the
      // dimensionality of the problem.
      distributed_table_ = &distributed_table_in;
      n_attributes_ = distributed_table_in.n_attributes();
      sampling_rate_ = sampling_rate_in;
    }

    /** @brief Reshuffles the data and builds the global top tree with
     *         the local trees.
     */
    template<typename MetricType>
    void Build(
      boost::mpi::communicator &world,
      const MetricType &metric_in, int leaf_size) {

      // The timer for building the global tree.
      boost::mpi::timer distributed_table_index_timer;

      // Start reshuffling.
      RecursiveReshuffle_(world, metric_in);

      // Given the reshuffled data, build local trees on each data.
      world.barrier();

      // Report timing for the master process.
      if(world.rank() == 0) {
        printf("Finished building the distributed tree.\n");
        printf(
          "Took %g seconds to read in the distributed tree.\n",
          distributed_table_index_timer.elapsed());
        printf(
          "The following is the distribution of points among all MPI "
          "processes.\n");
        for(int i = 0; i < world.size(); i++) {
          printf(
            "Process %d has %d points.\n", i,
            distributed_table_->local_n_entries(i));
        }
      }
    }
};
}
}

#endif
