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
#include "core/parallel/distributed_tree_util.h"
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

  private:

    /** @brief The pointer to the distributed table. The data will be
     *         reshuffled and exchanged among the MPI processes.
     */
    DistributedTableType *distributed_table_;

    /** @brief The dimensionality.
     */
    int n_attributes_;

  private:

    /** @brief Recursively splits a given node creating its children.
     */
    template<typename MetricType>
    void RecursiveReshuffle_(
      boost::mpi::communicator &world,
      const MetricType &metric_in) {

      // If the communicator size is greater than one, then try to split.
      boost::mpi::communicator current_comm = world;
      while(current_comm.size() > 1) {

        // Find the bounding primitive containing all the points
        // belonging to the process belonging to the communicator.
        BoundType bound;
        bound.Init(n_attributes_);
        TreeSpecType::FindBoundFromMatrix(
          current_comm, metric_in,
          distributed_table_->local_table()->data(), &bound);

        // Find the split.
        std::vector< std::vector<int> > assigned_point_indices;
        std::vector<int> membership_counts_per_process;
        bool can_cut =
          TreeSpecType::AttemptSplitting(
            current_comm, metric_in, bound,
            distributed_table_->local_table()->data(),
            &assigned_point_indices, &membership_counts_per_process);

        if(can_cut) {

          // Reshuffle points among the processes.
          core::parallel::DistributedTreeUtil <
          DistributedTableType >::ReshufflePoints(
            current_comm, assigned_point_indices,
            membership_counts_per_process, distributed_table_, n_attributes_);

          // Split the communicator into two groups here and recurse.
          bool color = (current_comm.rank() < current_comm.size() / 2);
          current_comm = current_comm.split(color);
        }
        else {
          break;
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

    /** @brief Initialize with a given distributed table.
     *
     *  @param distributed_table_in The distributed table.
     */
    void Init(
      DistributedTableType &distributed_table_in) {

      // This assumes that each distributed table knows the
      // dimensionality of the problem.
      distributed_table_ = &distributed_table_in;
      n_attributes_ = distributed_table_in.n_attributes();
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

      // Refresh the final count on each distributed table on each
      // process.
      distributed_table_->RefreshCounts_(world);

      // Index the local tree on each process.
      distributed_table_->local_table()->IndexData(metric_in, leaf_size);

      // Build the top tree from the collected root nodes from all
      // processes.
      distributed_table_->BuildGlobalTree_(world, metric_in);

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
