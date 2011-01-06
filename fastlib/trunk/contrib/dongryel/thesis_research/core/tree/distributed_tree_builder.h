/** @file distributed_tree_builder.h
 *
 *  The generic template for building a distributed tree.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_DISTRIBUTED_TREE_BUILDER_H
#define CORE_TREE_DISTRIBUTED_TREE_BUILDER_H

#include <algorithm>
#include <boost/mpi.hpp>
#include "core/parallel/parallel_sample_sort.h"
#include "core/table/offset_dense_matrix.h"

namespace core {
namespace tree {
template<typename DistributedTableType>
class DistributedTreeBuilder {
  public:
    typedef typename DistributedTableType::TableType TableType;

  private:
    DistributedTableType *distributed_table_;

    double sampling_rate_;

  private:

    void SetupGatherPointers_(
      TableType &sampled_table, const std::vector<int> &counts,
      std::vector<core::table::SampleDenseMatrix> *gather_pointers_out) {

      // Have the pointers point to the right position based on the
      // prefix sum position.
      gather_pointers_out->resize(counts.size());
      int starting_column_index = 0;
      for(unsigned int i = 0; i < counts.size(); i++) {
        (*gather_pointers_out)[i].Init(
          sampled_table.data(), starting_column_index, counts[i]);
        starting_column_index += counts[i];
      }
    }

    void BuildSampleTree_(boost::mpi::communicator &world) {

      // Each process generates a random subset of the data points to
      // send to the master. This is a MPI gather operation.
      TableType sampled_table;
      std::vector<int> sampled_indices;
      SelectSubset_(sampling_rate_, &sampled_indices);

      // Send the number of points chosen in this process to the
      // master so that the master can allocate the appropriate amount
      // of space to receive all the points.
      std::vector<int> counts;
      int local_sampled_indices_size = static_cast<int>(
                                         sampled_indices.size());
      boost::mpi::gather(world, local_sampled_indices_size, counts, 0);

      // The master process allocates the sample table and gathers the
      // chosen samples from each process.
      core::table::SampleDenseMatrix local_pointer;
      local_pointer.Init(
        distributed_table_->local_table()->data(), sampled_indices);
      if(world.rank() == 0) {
        int total_num_samples = std::accumulate(
                                  counts.begin(), counts.end(), 0);
        sampled_table.Init(
          distributed_table_->n_attributes(), total_num_samples);
        std::vector<core::table::SampleDenseMatrix> gather_pointers;
        SetupGatherPointers_(sampled_table, counts, gather_pointers);
      }
      boost::mpi::gather(world, local_pointer, gather_pointers, 0);
    }

    void SelectSubset_(
      std::vector<int> *sampled_indices_out) {

      std::vector<int> indices(owned_table_->n_entries(), 0);
      for(unsigned int i = 0; i < indices.size(); i++) {
        indices[i] = i;
      }
      int num_elements =
        std::max(
          (int) floor(sampling_rate_ * owned_table_->n_entries()), 1);
      std::random_shuffle(indices.begin(), indices.end());

      for(int i = 0; i < num_elements; i++) {
        sampled_indices_out->push_back(indices[i]);
      }
    }

  public:
    void Init(
      DistributedTableType &distributed_table_in, double sampling_rate_in) {
      distributed_table_ = &distributed_table_in;
      sampling_rate_ = sampling_rate_in;
    }

    void BuildTree(boost::mpi::communicator &world) {
    }
};
};
};

#endif
