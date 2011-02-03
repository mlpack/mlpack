/** @file sub_table.test.cc
 *
 *  A "stress" test driver for sub tables.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/test/unit_test.hpp>
#include "core/metric_kernels/lmetric.h"
#include "core/table/sub_table.h"
#include "core/table/sub_table_list.h"
#include "core/table/distributed_table.h"
#include "core/table/table.h"
#include "core/tree/gen_kdtree.h"
#include "core/tree/gen_metric_tree.h"
#include "core/math/math_lib.h"
#include <time.h>

namespace core {
namespace table {

template<typename SubTableListType>
class TestSubTable {

  public:

    typedef typename SubTableListType::SubTableType SubTableType;

    typedef typename SubTableType::TableType TableType;

    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

    typedef typename TableType::TreeType TreeType;

    typedef core::table::DistributedTable <
    typename TreeType::TreeSpecType > DistributedTableType;

  private:

    class TreeNodeListType {
      private:

        // For boost serialization.
        friend class boost::serialization::access;

        std::vector<TreeType *> vector_;

        bool is_alias_;

      public:

        int size() const {
          return static_cast<int>(vector_.size());
        }

        void Init(const std::vector<TreeType *> &vector_in) {
          is_alias_ = true;
          vector_ = vector_in;
        }

        ~TreeNodeListType() {
          if(! is_alias_) {
            for(unsigned int i = 0; i < vector_.size(); i++) {
              delete vector_[i];
            }
          }
        }

        TreeType *operator[](int index) {
          return vector_[index];
        }

        std::vector<TreeType *> &vector() {
          return vector_;
        }

        template<class Archive>
        void save(Archive &ar, const unsigned int version) const {
          int size = vector_.size();
          ar & size;
          for(int i = 0; i < size; i++) {
            ar & vector_[i];
          }
        }

        template<class Archive>
        void load(Archive &ar, const unsigned int version) {
          is_alias_ = false;
          int size;
          ar & size;
          vector_.resize(size);
          for(int i = 0; i < size; i++) {
            ar & vector_[i];
          }
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()
    };

    void GenerateRandomDataset_(
      int num_dimensions,
      int num_points,
      TableType *random_dataset) {

      random_dataset->Init(num_dimensions, num_points);

      for(int j = 0; j < num_points; j++) {
        core::table::DensePoint point;
        random_dataset->get(j, &point);
        for(int i = 0; i < num_dimensions; i++) {
          point[i] = core::math::Random(0.1, 1.0);
        }
      }
    }

    void CheckIntegrity_(TreeType *node) {
      if(node->begin() < 0 || node->count() < 0) {
        printf("The node begin/count is messed up.\n");
        exit(0);
      }

      if(! node->is_leaf()) {
        CheckIntegrity_(node->left());
        CheckIntegrity_(node->right());
      }
    }

  public:

    int StressTestMain(boost::mpi::communicator &world) {
      for(int i = 0; i < 10; i++) {
        int num_dimensions;
        if(world.rank() == 0) {
          num_dimensions = core::math::RandInt(3, 20);
        }
        boost::mpi::broadcast(world, num_dimensions, 0);
        int num_points = core::math::RandInt(3000, 5001);
        if(StressTest(world, num_dimensions, num_points) == false) {
          printf("Failed!\n");
          exit(0);
        }
      }
      return 0;
    }

    bool StressTest(
      boost::mpi::communicator &world, int num_dimensions, int num_points) {

      std::cout << "Number of dimensions: " << num_dimensions << "\n";
      std::cout << "Number of points: " << num_points << "\n";

      // Push in the reference dataset name.
      std::stringstream reference_file_name_sstr;
      reference_file_name_sstr << "random_dataset.csv" << world.rank();
      std::string references_in = reference_file_name_sstr.str();

      // Generate the random dataset and save it.
      {
        TableType random_table;
        GenerateRandomDataset_(
          num_dimensions, num_points, &random_table);
        random_table.Save(references_in);
      }

      // Generate the leaf size.
      int leaf_size;
      if(world.rank() == 0) {
        leaf_size = core::math::RandInt(15, 25);
        printf("Choosing the leaf size of %d\n", leaf_size);
      }
      boost::mpi::broadcast(world, leaf_size, 0);

      // Generate the sample probability.
      double sample_probability;
      if(world.rank() == 0) {
        sample_probability = core::math::Random(0.2, 0.5);
        printf("Choosing the sample probability of %g\n", sample_probability);
      }
      boost::mpi::broadcast(world, sample_probability, 0);

      // Read the distributed table.
      DistributedTableType distributed_table;
      distributed_table.Init(references_in, world);
      core::metric_kernels::LMetric<2> l2_metric;
      distributed_table.IndexData(
        l2_metric, world, leaf_size, sample_probability);

      // Get the list of all nodes.
      TreeNodeListType local_list_of_nodes;
      distributed_table.local_table()->get_nodes(
        distributed_table.local_table()->get_tree(),
        &(local_list_of_nodes.vector()));
      std::vector< TreeNodeListType > global_lists;

      // Do an all gather of all the nodes.
      boost::mpi::all_gather(world, local_list_of_nodes, global_lists);

      // Repeat the all-to-all exchange 10 times.
      for(int i = 0; i < 10; i++) {

        // The master thinks of how many levels of trees to serialize.
        int num_levels_to_serialize;
        if(world.rank() == 0) {
          num_levels_to_serialize = core::math::RandInt(2, 10);
        }
        boost::mpi::broadcast(world, num_levels_to_serialize, 0);

        // Define cache block size naively.
        int cache_block_size = (1 << num_levels_to_serialize) * leaf_size;

        // The master thinks of how many subtables to exchange.
        int num_subtables_to_exchange;
        if(world.rank() == 0) {
          num_subtables_to_exchange = core::math::RandInt(2, 10);
        }
        boost::mpi::broadcast(world, num_subtables_to_exchange, 0);

        std::vector< std::vector< core::table::DenseMatrix> > data_matrices(
          world.size());
        std::vector< std::vector< OldFromNewIndexType * > > old_from_news(
          world.size());

        for(int j = 0; j < world.size(); j++) {
          if(j != world.rank()) {
            data_matrices[j].resize(num_subtables_to_exchange);
            old_from_news[j].resize(num_subtables_to_exchange);
            for(int k = 0; k < num_subtables_to_exchange; k++) {
              data_matrices[j][k].Init(
                distributed_table.n_attributes(), cache_block_size);
              old_from_news[j][k] =
                new OldFromNewIndexType[ cache_block_size ];
            }
          }
        }

        // Prepare the list of subtables, and do another all_to_all.
        std::vector< SubTableListType > send_subtables(world.size());
        for(int j = 0; j < world.size(); j++) {
          if(j != world.rank()) {
            for(int i = 0; i < num_subtables_to_exchange; i++) {

              // Pick a random node.
              int random_node_id =
                core::math::RandInt(0, local_list_of_nodes.size());
              int begin = local_list_of_nodes[random_node_id]->begin();
              int count = local_list_of_nodes[random_node_id]->count();
              send_subtables[j].push_back(
                distributed_table.local_table(),
                distributed_table.local_table()->get_tree()->
                FindByBeginCount(begin, count), num_levels_to_serialize);
            }
          }
        }

        // Prepare the subtable list to be received. Right now, we
        // just receive one subtable per process.
        std::vector< SubTableListType > received_subtables_in_this_round;
        received_subtables_in_this_round.resize(world.size());
        for(int j = 0; j < world.size(); j++) {
          if(j != world.rank()) {
            for(int i = 0; i < num_subtables_to_exchange; i++) {
              // Allocate the cache block to the subtable that is about
              // to be received.
              received_subtables_in_this_round[j].push_back(
                j, data_matrices[j][i], old_from_news[j][i],
                0, cache_block_size, num_levels_to_serialize);
            }
          }
        }

        // All-to-all to exchange the subtables.
        boost::mpi::all_to_all(
          world, send_subtables, received_subtables_in_this_round);

        // Check integrity.
        for(unsigned int i = 0;
            i < received_subtables_in_this_round.size(); i++) {
          for(unsigned int j = 0;
              j < received_subtables_in_this_round[j].size(); j++) {
            CheckIntegrity_(
              received_subtables_in_this_round[j][i].table()->get_tree());
          }
        }

        // Free memory.
        for(unsigned int i = 0; i < old_from_news.size(); i++) {
          for(unsigned int j = 0; j < old_from_news[i].size(); j++) {
            delete[] old_from_news[i][j];
          }
        }
      } // end of the trial loop.

      return true;
    }
};
}
}

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  core::math::global_random_number_state_.set_seed(time(NULL) + world.rank());

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<core::tree::AbstractStatistic>,
       std::pair<int, std::pair<int, int> > > TableType;
  typedef core::table::SubTable<TableType> SubTableType;
  typedef core::table::SubTableList<SubTableType> SubTableListType;

  // Call the tests.
  core::table::TestSubTable<SubTableListType> sub_table_test;
  sub_table_test.StressTestMain(world);
  std::cout << "All tests passed!\n";

  return 0;
}
