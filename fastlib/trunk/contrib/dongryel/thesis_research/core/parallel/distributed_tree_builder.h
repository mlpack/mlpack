/** @file distributed_tree_builder.h
 *
 *  The generic template for building a distributed tree.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_TREE_BUILDER_H
#define CORE_PARALLEL_DISTRIBUTED_TREE_BUILDER_H

#include <algorithm>
#include <numeric>
#include <boost/scoped_array.hpp>
#include <boost/bind.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include "core/parallel/parallel_sample_sort.h"
#include "core/table/offset_dense_matrix.h"
#include "core/table/memory_mapped_file.h"
#include "core/tree/hrect_bound.h"

namespace core {
namespace parallel {
class RangeCombine:
  public std::binary_function <
    core::math::Range, core::math::Range, core::math::Range > {
  public:
    const core::math::Range operator()(
      const core::math::Range &a, const core::math::Range &b) const {
      core::math::Range return_range;
      return_range |= a;
      return_range |= b;
      return return_range;
    }
};
}
}

namespace boost {
namespace mpi {
template<>
class is_commutative <
  core::parallel::RangeCombine, core::math::Range > :
  public boost::mpl::true_ {

};
}
}

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace core {
namespace parallel {
template<typename DistributedTableType>
class DistributedTreeBuilder {

  public:
    typedef typename DistributedTableType::TableType TableType;

    typedef typename DistributedTableType::TreeType TreeType;

    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

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

    /** @brief The sampling rate at which to send to the master for
     *         building a top tree.
     */
    double sampling_rate_;

  private:

    static bool MortonOrderNodes_(TreeType *first_node, TreeType *second_node) {
      return
        core::math::MortonOrderPoints(
          first_node->bound().center(), second_node->bound().center());
    }

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

    /** @brief Augment the list of nodes so that it matches the number
     *         of processes for convenience.
     *
     *  @param world The communicator.
     *  @param global_root_bound The global bound of all the points across all
     *         processes.
     *  @param top_leaf_nodes The current list of sample tree leaf nodes.
     *         It will be augmented an additional number of nodes to match
     *         the number of processes.
     */
    void AugmentNodes_(
      boost::mpi::communicator &world,
      const core::tree::HrectBound &global_root_bound,
      std::vector<TreeType *> &top_leaf_nodes) {

      core::table::DensePoint tmp_point;
      arma::vec tmp_point_alias;
      tmp_point.Init(top_leaf_nodes[0]->bound().center().length());
      core::table::DensePointToArmaVec(tmp_point, &tmp_point_alias);
      int num_additional = world.size() - top_leaf_nodes.size();
      int num_samples = std::max(1, core::math::RandInt(top_leaf_nodes.size()));

      // Randomly add new dummy nodes with randomly chosen centroids
      // averaged.
      for(int j = 0; j < num_additional; j++) {
        global_root_bound.RandomPointInside(&tmp_point_alias);
        for(int i = 0; i < num_samples; i++) {
          arma::vec random_node_center;
          core::table::DensePointToArmaVec(
            top_leaf_nodes[
              core::math::RandInt(top_leaf_nodes.size())]->bound().center(),
            &random_node_center);
          tmp_point_alias += random_node_center;
        }
        tmp_point_alias = (1.0 / static_cast<double>(num_samples)) *
                          tmp_point_alias;
        top_leaf_nodes.push_back(new TreeType());
        top_leaf_nodes[ top_leaf_nodes.size() - 1 ]->bound().center().Copy(
          tmp_point);
      }
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

    template<typename MetricType>
    void GetLeafNodeMembershipCounts_(
      const MetricType &metric_in,
      const std::vector<TreeType *> &top_leaf_nodes,
      std::vector< std::vector<int> > *assigned_point_indices,
      std::vector<int> *membership_counts_per_node) {

      assigned_point_indices->resize(0);
      assigned_point_indices->resize(top_leaf_nodes.size());
      membership_counts_per_node->resize(top_leaf_nodes.size());
      std::fill(
        membership_counts_per_node->begin(),
        membership_counts_per_node->end(), 0.0);

      // Loop through each point and find the closest leaf node.
      for(int i = 0; i < distributed_table_->local_table()->n_entries(); i++) {
        core::table::DensePoint point;
        distributed_table_->local_table()->get(i, &point);

        // Loop through each leaf node.
        double min_squared_mid_distance = std::numeric_limits<double>::max();
        int min_index = -1;
        for(unsigned int j = 0; j < top_leaf_nodes.size(); j++) {
          const typename TreeType::BoundType &leaf_node_bound =
            top_leaf_nodes[j]->bound();

          // Compute the squared mid-distance.
          double squared_mid_distance = leaf_node_bound.MidDistanceSq(
                                          metric_in, point);
          if(squared_mid_distance < min_squared_mid_distance) {
            min_squared_mid_distance = squared_mid_distance;
            min_index = j;
          }
        }

        // Output the assignments.
        (*assigned_point_indices)[min_index].push_back(i);
        (*membership_counts_per_node)[min_index]++;
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

    /** @brief Greedily assign each point on the current process to
     *         the closest node among the list and re-distribute.
     */
    template<typename MetricType>
    void GreedyAssign_(
      boost::mpi::communicator &world,
      const MetricType &metric_in,
      const std::vector<TreeType *> &top_leaf_nodes) {

      // Determine the membership counts.
      std::vector< std::vector<int> > assigned_point_indices;
      std::vector<int> membership_counts_per_node;

      GetLeafNodeMembershipCounts_(
        metric_in, top_leaf_nodes,
        &assigned_point_indices, &membership_counts_per_node);

      ReshufflePoints_(
        world, assigned_point_indices, membership_counts_per_node);
    }

    template<typename MetricType>
    void BuildSampleTree_(
      boost::mpi::communicator &world,
      const MetricType &metric_in,
      const core::tree::HrectBound &global_root_bound,
      TableType *sampled_table_out,
      std::vector<TreeType *> *top_leaf_nodes_out) {

      // Each process generates a random subset of the data points to
      // send to the master. This is a MPI gather operation.
      std::vector<int> sampled_indices;
      SelectSubset_(&sampled_indices);

      // Send the number of points chosen in this process to the
      // master so that the master can allocate the appropriate amount
      // of space to receive all the points.
      std::vector<int> counts;
      int local_sampled_indices_size = static_cast<int>(
                                         sampled_indices.size());
      boost::mpi::gather(world, local_sampled_indices_size, counts, 0);

      // The master process allocates the sample table and gathers the
      // chosen samples from each process.
      SampleDenseMatrixType local_pointer;
      local_pointer.Init(
        distributed_table_->local_table()->data(),
        distributed_table_->local_table()->old_from_new(),
        sampled_indices);
      std::vector<SampleDenseMatrixType> gather_pointers;
      if(world.rank() == 0) {
        int total_num_samples = std::accumulate(
                                  counts.begin(), counts.end(), 0);
        sampled_table_out->Init(
          n_attributes_, total_num_samples);
        SetupGatherPointers_(*sampled_table_out, counts, &gather_pointers);

        // The master process actually needs to setup its own portion
        // manually since MPI gather does not call
        // serialize/unserialize on the self.
        local_pointer.Export(
          gather_pointers[0].matrix(),
          gather_pointers[0].old_from_new(),
          gather_pointers[0].starting_column_index());
      }
      boost::mpi::gather(world, local_pointer, gather_pointers, 0);

      // Index the tree up so that the number of leaf nodes matches
      // the number of processes. If missing some nodes, then sample a
      // region and try to make up a node.
      if(world.rank() == 0) {
        top_leaf_nodes_out->resize(0);
        sampled_table_out->IndexData(metric_in, 1, world.size());
        sampled_table_out->get_leaf_nodes(
          sampled_table_out->get_tree(), top_leaf_nodes_out);
        if(top_leaf_nodes_out->size() <
            static_cast<unsigned int>(world.size())) {
          AugmentNodes_(
            world, global_root_bound, *top_leaf_nodes_out);
        }

        // Sort the nodes by Z-ordering their centroids.
        std::sort(
          top_leaf_nodes_out->begin(), top_leaf_nodes_out->end(),
          MortonOrderNodes_);
      }
      boost::mpi::broadcast(world, *top_leaf_nodes_out, 0);
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

    /** @brief Compute the centroid of the points owned by the current
     *         process and sort them in increasing order of distance
     *         to the centroid.
     */
    template<typename MetricType>
    void RankPointsFromItsCentroid_(
      const MetricType &metric_in,
      std::vector<int> *sorted_indices_increasing) {

      // First compute the centroid.
      arma::vec centroid;
      centroid.zeros(n_attributes_);
      for(int i = 0; i < distributed_table_->local_table()->n_entries(); i++) {
        arma::vec point;
        distributed_table_->local_table()->get(i, &point);
        centroid += point;
      }
      centroid /=
        static_cast<double>(distributed_table_->local_table()->n_entries());

      // Pairs of point id and its squared distance from the centroid
      // and sort them.
      std::vector< std::pair<int, double> > point_id_distance_pairs;
      point_id_distance_pairs.resize(
        distributed_table_->local_table()->n_entries());
      for(int i = 0; i < distributed_table_->local_table()->n_entries(); i++) {
        arma::vec point;
        distributed_table_->local_table()->get(i, &point);
        point_id_distance_pairs[i].first = i;
        point_id_distance_pairs[i].second =
          metric_in.DistanceSq(point, centroid);
      }
      std::sort(
        point_id_distance_pairs.begin(), point_id_distance_pairs.end(),
        boost::bind(&std::pair<int, double>::second, _1) <
        boost::bind(&std::pair<int, double>::second, _2));
    }

    /** @brief Subsample a list of indices from the locally owned table.
     */
    void SelectSubset_(
      std::vector<int> *sampled_indices_out) {

      std::vector<int> indices(
        distributed_table_->local_table()->n_entries(), 0);
      for(unsigned int i = 0; i < indices.size(); i++) {
        indices[i] = i;
      }
      int num_elements =
        std::max(
          (int)
          floor(
            sampling_rate_ *
            distributed_table_->local_table()->n_entries()), 1);
      std::random_shuffle(indices.begin(), indices.end());

      for(int i = 0; i < num_elements; i++) {
        sampled_indices_out->push_back(indices[i]);
      }
    }

  public:

    /** @brief The default constructor.
     */
    DistributedTreeBuilder() {
      distributed_table_ = NULL;
      n_attributes_ = 0;
      sampling_rate_ = 0;
    }

    /** @brief Initialize with a given distributed table with the
     *         given sampling rate for building the top tree.
     *
     *  @param distributed_table_in The distributed table.
     *  @param sampling_rate_in The sampling rate used for building a top
     *         sample tree.
     */
    void Init(
      DistributedTableType &distributed_table_in, double sampling_rate_in) {

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

      // Timer for the tree building.
      boost::mpi::timer distributed_table_index_timer;

      // Do a reduction to find the rough global bound of all the
      // points across all processes.
      boost::scoped_array<core::math::Range> global_root_bound_vector(
        new core::math::Range[n_attributes_]);
      boost::scoped_array<core::math::Range> local_bound(
        new core::math::Range[n_attributes_]);
      for(int i = 0; i < distributed_table_->local_table()->n_entries(); i++) {
        core::table::DensePoint point;
        distributed_table_->local_table()->get(i, &point);
        for(int d = 0; d < n_attributes_; d++) {
          local_bound[d] |= point[d];
        }
      }
      boost::mpi::all_reduce(
        world, local_bound.get(), n_attributes_,
        global_root_bound_vector.get(), core::parallel::RangeCombine());
      core::tree::HrectBound global_root_bound;
      global_root_bound.Init(n_attributes_);
      for(int i = 0; i < n_attributes_; i++) {
        global_root_bound.get(i) = global_root_bound_vector[i];
      }

      for(int num_outer_it = 0; num_outer_it < 3; num_outer_it++) {

        // Build the initial sample tree.
        std::vector<TreeType *> top_leaf_nodes;
        TableType sampled_table;
        BuildSampleTree_(
          world, metric_in, global_root_bound, &sampled_table, &top_leaf_nodes);

        // For each process, determine the membership of each points to
        // each of the top leaf nodes and do an all-to-all to do the
        // reshuffle.
        GreedyAssign_(world, metric_in, top_leaf_nodes);

        // Recompute the centroids of each process and sort each point
        // according to its distance from its centroid.
        std::vector<int> sorted_indices_increasing;
        RankPointsFromItsCentroid_(metric_in, &sorted_indices_increasing);

        // Do a re-distribution of points.
        RedistributeEqually_(world, metric_in, sorted_indices_increasing);

        // Destroy the top leaf nodes, if not the master process.
        if(world.rank() != 0) {
          for(unsigned int i = 0; i < top_leaf_nodes.size(); i++) {
            delete top_leaf_nodes[i];
          }
        }

        // Barrier.
        world.barrier();

      } // end of the iterative process of reshuffling.

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
      }
    }
};
}
}

#endif
