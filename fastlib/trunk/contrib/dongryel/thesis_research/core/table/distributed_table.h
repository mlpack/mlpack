/** @file distributed_table.h
 *
 *  An abstract class for maintaining a distributed set of points.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_TABLE_H
#define CORE_TABLE_DISTRIBUTED_TABLE_H

#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/string.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/utility.hpp>
#include "core/parallel/sample_distributed_tree_builder.h"
#include "core/parallel/vanilla_distributed_tree_builder.h"
#include "core/table/index_util.h"
#include "core/table/memory_mapped_file.h"
#include "core/table/table.h"
#include "core/tree/gen_metric_tree.h"

namespace core {
namespace table {

extern MemoryMappedFile *global_m_file_;

template<typename IncomingTreeSpecType>
class DistributedTable: public boost::noncopyable {

  public:

    /** @brief The type of the specification used to index the data.
     */
    typedef IncomingTreeSpecType TreeSpecType;

    /** @brief The type of the tree used to index data.
     */
    typedef core::tree::GeneralBinarySpaceTree <TreeSpecType> TreeType;

    /** @brief Defines the type of the local table owned by the
     *         distributed table.
     */
    typedef core::table::Table <
    TreeSpecType, std::pair<int, std::pair< int, int> > > TableType;

    /** @brief Defines the type of the distributed table.
     */
    typedef DistributedTable<TreeSpecType> DistributedTableType;

    /** @brief The index type for each point is a pair of a machine ID
     *         and a point ID.
     */
    typedef std::pair<int, std::pair<int, int> > IndexType;

    // For giving private access to the distributed tree builder class.
    friend class core::parallel::SampleDistributedTreeBuilder <
        DistributedTableType >;

  friend class core::parallel::VanillaDistributedTreeBuilder <
        DistributedTableType >;

  private:

    /** @brief The pointer to the local set of data owned by the
     *         process.
     */
    boost::interprocess::offset_ptr<TableType> owned_table_;

    /** @brief The global view of the number of points across all MPI
     *         processes.
     */
    boost::interprocess::offset_ptr<int> local_n_entries_;

    /** @brief The indexing of the top tree.
     */
    boost::interprocess::offset_ptr<TableType> global_table_;

    /** @brief The number of MPI processes present.
     */
    int world_size_;

  private:

    /** @brief Refresh the number of points from all of the active MPI
     *         processes.
     */
    void RefreshCounts_(boost::mpi::communicator &world) {
      boost::mpi::all_gather(
        world, owned_table_->n_entries(), local_n_entries_.get());
    }

    template<typename MetricType, typename BoundType>
    void AdjustBounds_(
      const MetricType &metric,
      const std::vector<BoundType> &bounds, TreeType *node) {

      if(node->is_leaf()) {
        typename TableType::TreeIterator node_it =
          global_table_->get_node_iterator(node);
        int process_id;
        node_it.Next(&process_id);
        node->bound().Copy(bounds[process_id]);
        while(node_it.HasNext()) {
          int process_id;
          node_it.Next(&process_id);
          node->bound().Expand(metric, bounds[process_id]);
        }
      }
      else {

        // Adjust so that the node bound contains the bounds of the
        // children.
        AdjustBounds_(metric, bounds, node->left());
        AdjustBounds_(metric, bounds, node->right());
        node->bound().Copy(node->left()->bound());
        node->bound().Expand(metric, node->right()->bound());
      }
    }

    template<typename MetricType>
    void BuildGlobalTree_(
      boost::mpi::communicator &world, const MetricType &metric_in) {

      // Every process gathers the adjusted leaf centroids and build
      // the top tree individually.
      global_table_ =
        (core::table::global_m_file_) ?
        core::table::global_m_file_->Construct<TableType>() : new TableType();
      if(world.rank() == 0) {
        global_table_->Init(
          owned_table_->n_attributes(), world.size());
      }
      core::table::DensePoint root_bound_center;
      owned_table_->get_tree()->bound().center(&root_bound_center);
      boost::mpi::gather(
        world,
        root_bound_center.ptr(),
        owned_table_->n_attributes(), global_table_->data().ptr(), 0);

      // The master builds the tree and broadcasts the nodes.
      if(world.rank() == 0) {
        global_table_->IndexData(metric_in, 1);
      }

      // After building the tree, all processes send the root bound
      // primitive to the master process.
      std::vector<typename TreeType::BoundType> bounds;
      boost::mpi::gather(
        world, owned_table_->get_tree()->bound(), bounds, 0);
      if(world.rank() == 0) {
        AdjustBounds_(metric_in, bounds, global_table_->get_tree());
      }

      // Broadcast the global tree.
      boost::mpi::broadcast(world, *global_table_, 0);
    }

  public:

    /** @brief The default constructor.
     */
    DistributedTable() {
      owned_table_ = NULL;
      local_n_entries_ = NULL;
      global_table_ = NULL;
      world_size_ = -1;
    }

    /** @brief Destructor.
     */
    ~DistributedTable() {

      // Delete the list of number of entries for each table in the
      // distributed table.
      if(local_n_entries_.get() != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(local_n_entries_.get());
        }
        else {
          delete[] local_n_entries_.get();
        }
        local_n_entries_ = NULL;
      }

      // Delete the table.
      if(owned_table_.get() != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(owned_table_.get());
        }
        else {
          delete owned_table_.get();
        }
        owned_table_ = NULL;
      }

      // Delete the tree.
      if(global_table_.get() != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(global_table_.get());
        }
        else {
          delete global_table_.get();
        }
        global_table_ = NULL;
      }
    }

    /** @brief Destroys a pre-existing locally owned table and sets it
     *         to a new one.
     */
    void set_local_table(TableType *new_local_table_in) {
      if(owned_table_.get() != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(owned_table_.get());
        }
        else {
          delete owned_table_.get();
        }
      }
      owned_table_ = new_local_table_in;
    }

    /** @brief Retrieves the local table owned by the distributed
     *         table object.
     */
    TableType *local_table() {
      return owned_table_.get();
    }

    /** @brief Retrieves the global old from new mapping.
     */
    typename TableType::OldFromNewIndexType *old_from_new() {
      return global_table_->old_from_new();
    }

    /** @brief Retrieves the global new from old mapping.
     */
    int *new_from_old() {
      return global_table_->new_from_old();
    }

    /** @brief Retrieves the global tree.
     */
    TreeType *get_tree() {
      return global_table_->get_tree();
    }

    /** @brief Retrieves the dimensionality of the table.
     */
    int n_attributes() const {
      return owned_table_->n_attributes();
    }

    /** @brief Looks up points owned by any MPI process.
     */
    int local_n_entries(int rank_in) const {
      if(rank_in >= world_size_) {
        printf(
          "Invalid rank specified: %d. %d is the limit.\n",
          rank_in, world_size_);
        return -1;
      }
      return local_n_entries_[rank_in];
    }

    /** @brief Retrieves the number of points owned locally.
     */
    int n_entries() const {
      return owned_table_->n_entries();
    }

    /** @brief Initializes a distributed table, reading in the local
     *         set of the data owned by the process. This is
     *         coordinated across all MPI processes.
     */
    void Init(
      const std::string & file_name,
      boost::mpi::communicator &world,
      const std::string *weight_file_name = NULL) {

      boost::mpi::timer distributed_table_init_timer;

      // Initialize the table owned by the distributed table.
      owned_table_ = (core::table::global_m_file_) ?
                     core::table::global_m_file_->Construct<TableType>() :
                     new TableType();
      owned_table_->Init(file_name, world.rank(), weight_file_name);

      // Allocate the vector for storing the number of entries for all
      // the tables in the world, and do an all-gather operation to
      // find out all the sizes.
      world_size_ = world.size();
      local_n_entries_ = (core::table::global_m_file_) ?
                         (int *) global_m_file_->ConstructArray<int>(
                           world.size()) :
                         new int[ world.size()];
      boost::mpi::all_gather(
        world, owned_table_->n_entries(),
        local_n_entries_.get());

      if(world.rank() == 0) {
        printf(
          "Took %g seconds to read in the distributed tables.\n",
          distributed_table_init_timer.elapsed());
      }
    }

    /** @brief Returns whether the global tree has been built already.
     */
    bool IsIndexed() const {
      return global_table_->get_tree() != NULL;
    }

    /** @brief Builds a distributed tree.
     */
    template<typename MetricType>
    void IndexData(
      const MetricType & metric_in,
      boost::mpi::communicator &world,
      int leaf_size, double sample_probability_in) {

      core::parallel::VanillaDistributedTreeBuilder <
      DistributedTableType > builder;
      builder.Init(*this);
      builder.Build(world, metric_in, leaf_size);
    }

    typename TableType::TreeIterator get_node_iterator(TreeType *node) {
      return global_table_->get_node_iterator(node);
    }

    typename TableType::TreeIterator get_node_iterator(int begin, int count) {
      return global_table_->get_node_iterator(begin, count);
    }
};
}
}

#endif
