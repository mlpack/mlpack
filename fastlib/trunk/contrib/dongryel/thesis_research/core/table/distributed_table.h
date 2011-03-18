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
    typedef std::pair<int, int> IndexType;

    // For giving private access to the distributed tree builder class.
    friend class core::parallel::SampleDistributedTreeBuilder <
        DistributedTableType >;
  friend class core::parallel::VanillaDistributedTreeBuilder <
        DistributedTableType >;

  public:
    class TreeIterator {
      private:
        int begin_;

        int end_;

        int current_index_;

        const DistributedTableType *table_;

      public:

        TreeIterator() {
          begin_ = -1;
          end_ = -1;
          current_index_ = -1;
          table_ = NULL;
        }

        TreeIterator(const TreeIterator &it_in) {
          begin_ = it_in.begin();
          end_ = it_in.end();
          current_index_ = it_in.current_index();
          table_ = it_in.table();
        }

        TreeIterator(const DistributedTableType &table, const TreeType *node) {
          table_ = &table;
          begin_ = node->begin();
          end_ = node->end();
          current_index_ = begin_ - 1;
        }

        TreeIterator(const DistributedTableType &table, int begin, int count) {
          table_ = &table;
          begin_ = begin;
          end_ = begin + count;
          current_index_ = begin_ - 1;
        }

        const DistributedTableType *table() const {
          return table_;
        }

        bool HasNext() const {
          return current_index_ < end_ - 1;
        }

        void Next() {
          current_index_++;
        }

        void Next(core::table::DensePoint *entry, int *point_id) {
          current_index_++;
          table_->iterator_get_(current_index_, entry);
          *point_id = table_->iterator_get_id_(current_index_);
        }

        void get(int i, core::table::DensePoint *entry) {
          table_->iterator_get_(begin_ + i, entry);
        }

        void get_id(int i, int *point_id) {
          *point_id = table_->iterator_get_id_(begin_ + i);
        }

        void RandomPick(core::table::DensePoint *entry) {
          table_->iterator_get_(core::math::Random(begin_, end_), entry);
        }

        void RandomPick(core::table::DensePoint *entry, int *point_id) {
          *point_id = core::math::Random(begin_, end_);
          table_->iterator_get_(*point_id, entry);
        }

        void Reset() {
          current_index_ = begin_ - 1;
        }

        int current_index() const {
          return current_index_;
        }

        int count() const {
          return end_ - begin_;
        }

        int begin() const {
          return begin_;
        }

        int end() const {
          return end_;
        }
    };

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

      core::parallel::SampleDistributedTreeBuilder <
      DistributedTableType > builder;
      builder.Init(*this, sample_probability_in);
      builder.Build(world, metric_in, leaf_size);
    }

    TreeIterator get_node_iterator(TreeType *node) {
      return TreeIterator(*this, node);
    }

    TreeIterator get_node_iterator(int begin, int count) {
      return TreeIterator(*this, begin, count);
    }

  private:

    void direct_get_(int point_id, double *entry) const {
      if(this->IsIndexed() == false) {
        global_table_->data().MakeolumnVector(point_id, entry);
      }
      else {
        global_table_->data().MakeColumnVector(
          IndexUtil< IndexType>::Extract(
            global_table_->new_from_old(), point_id), entry);
      }
    }

    void direct_get_(
      int point_id, core::table::DensePoint *entry) const {
      if(this->IsIndexed() == false) {
        global_table_->data().MakeColumnVector(point_id, entry);
      }
      else {
        global_table_->data().MakeColumnVector(
          IndexUtil<IndexType>::Extract(
            global_table_->new_from_old(), point_id), entry);
      }
    }

    void direct_get_(int point_id, core::table::DensePoint *entry) {
      if(this->IsIndexed() == false) {
        global_table_->data().MakeColumnVector(point_id, entry);
      }
      else {
        global_table_->data().MakeColumnVector(
          IndexUtil<IndexType>::Extract(
            global_table_->new_from_old(), point_id), entry);
      }
    }

    void iterator_get_(
      int reordered_position, core::table::DensePoint *entry) const {
      global_table_->data().MakeColumnVector(reordered_position, entry);
    }

    void iterator_get_(
      int reordered_position, core::table::DensePoint *entry) {
      global_table_->data().MakeColumnVector(reordered_position, entry);
    }

    int iterator_get_id_(int reordered_position) const {
      if(this->IsIndexed() == false) {
        return reordered_position;
      }
      else {
        return IndexUtil<IndexType>::Extract(
                 global_table_->old_from_new(), reordered_position);
      }
    }
};
}
}

#endif
