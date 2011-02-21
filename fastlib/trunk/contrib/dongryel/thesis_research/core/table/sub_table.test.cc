/** @file sub_table.test.cc
 *
 *  A "stress" test driver for sub tables.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
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

    bool TestNodeIterator_(
      TableType &original_table,
      TreeType *node_from_original_table,
      TableType &sub_table,
      TreeType *node_from_sub_table) {

      // Get the iterator for both nodes.
      typename TableType::TreeIterator original_it =
        original_table.get_node_iterator(node_from_original_table);
      typename TableType::TreeIterator sub_it =
        sub_table.get_node_iterator(node_from_sub_table);

      // The result here.
      bool result = true;

      if(sub_table.points_available_underneath(node_from_sub_table)) {

        // Check whether the counts are the same.
        result = (original_it.count() == sub_it.count());

        // Loop through each iterator and compare.
        int reordered_id = node_from_sub_table->begin();
        while(result && sub_it.HasNext()) {
          core::table::DensePoint original_point;
          int original_point_id;
          core::table::DensePoint copy_point;
          int copy_point_id;
          original_it.Next(&original_point, &original_point_id);
          sub_it.Next(&copy_point, &copy_point_id);

          // Check the IDs.
          result = (original_point_id == copy_point_id);
          if(! result) {
            printf("The reordered index %d was translated to %d, but "
                   "it should be %d.\n", reordered_id, copy_point_id,
                   original_point_id);
          }
          reordered_id++;
        }
      }

      if(result && (! node_from_sub_table->is_leaf())) {
        bool left_result = TestNodeIterator_(
                             original_table, node_from_original_table->left(),
                             sub_table, node_from_sub_table->left());
        bool right_result = TestNodeIterator_(
                              original_table, node_from_original_table->right(),
                              sub_table, node_from_sub_table->right());
        result = left_result && right_result;
      }
      return result;
    }

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
      if(! node->bound().is_initialized()) {
        printf("The node bound is not initialized.\n");
        exit(0);
      }

      if(! node->is_leaf()) {
        CheckIntegrity_(node->left());
        CheckIntegrity_(node->right());
      }
    }

  public:

    int StressTestMain(boost::mpi::communicator &world) {
      /*
      for(int i = 0; i < 50; i++) {
        int num_dimensions;
        if(world.rank() == 0) {
          num_dimensions = core::math::RandInt(3, 20);
        }
        boost::mpi::broadcast(world, num_dimensions, 0);
        int num_points = core::math::RandInt(300, 501);
        if(StressTest(world, num_dimensions, num_points) == false) {
          printf("Failed!\n");
          exit(0);
        }
      }
      */

      // Now each process extracts its own subtable and checks the
      // iterator against the iterator of the original table.
      for(int i = 0; i < 50; i++) {
        int num_dimensions = core::math::RandInt(3, 20);
        int num_points = core::math::RandInt(300, 501);
        if(SelfStressTest(world, num_dimensions, num_points) == false) {
          printf("Process %d failed on %d points!\n", world.rank(),
                 num_points);
          break;
        }

        printf("%d-th iteration done for Process %d\n\n", i, world.rank());
      }

      return 0;
    }

    bool SelfStressTest(
      boost::mpi::communicator &world, int num_dimensions, int num_points) {

      // Generate the random table.
      TableType random_table;
      GenerateRandomDataset_(
        num_dimensions, num_points, &random_table);
      core::metric_kernels::LMetric<2> l2_metric;
      int leaf_size = core::math::RandInt(15, 25);
      random_table.IndexData(l2_metric, leaf_size);

      // Get the list of nodes.
      std::vector<TreeType *> local_list_of_nodes;
      random_table.get_nodes(
        random_table.get_tree(), &local_list_of_nodes);

      // Repeat the self-test 10 times.
      for(int k = 0; k < 10; k++) {

        // Think of how many levels of trees to serialize.
        int max_num_levels_to_serialize = 3;
        printf("Serializing %d levels.\n", max_num_levels_to_serialize);

        // The subtable to test.
        SubTableType subtable_to_save;
        SubTableType subtable_to_load;

        // Pick a random node to start and save it to a file, and
        // reload it.
        int random_node_id =
          core::math::RandInt(
            0, static_cast<int>(local_list_of_nodes.size()));
        int begin = local_list_of_nodes[random_node_id]->begin();
        int count = local_list_of_nodes[random_node_id]->count();
        printf("Process %d chose %d %d to test\n", world.rank(), begin, count);
        subtable_to_save.Init(
          &random_table,
          random_table.get_tree()->FindByBeginCount(begin, count),
          max_num_levels_to_serialize, false);
        subtable_to_load.Init(0, max_num_levels_to_serialize, false);

        // File name to test.
        std::stringstream test_file_name_sstr;
        test_file_name_sstr << "test_file.csv" << world.rank();
        {
          // Save.
          std::ofstream ofs(test_file_name_sstr.str().c_str());
          boost::archive::text_oarchive oa(ofs);
          oa << subtable_to_save;
        }
        {
          // Load.
          std::ifstream ifs(test_file_name_sstr.str().c_str());
          boost::archive::text_iarchive ia(ifs);
          ia >> subtable_to_load;
        }

        // Test the iterator.
        if(TestNodeIterator_(
              random_table,
              random_table.get_tree()->FindByBeginCount(begin, count),
              *(subtable_to_load.table()),
              subtable_to_load.table()->get_tree()) == false) {
          return false;
        }

      } // end of the trial loop.

      return true;
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
      std::vector<TreeType *> local_list_of_nodes;
      distributed_table.local_table()->get_nodes(
        distributed_table.local_table()->get_tree(),
        &local_list_of_nodes);

      // Repeat the all-to-all exchange 10 times.
      for(int k = 0; k < 10; k++) {

        // The master thinks of how many levels of trees to serialize.
        int max_num_levels_to_serialize;
        if(world.rank() == 0) {
          max_num_levels_to_serialize = core::math::RandInt(2, 10);
          printf("Serializing %d levels.\n", max_num_levels_to_serialize);
        }
        boost::mpi::broadcast(world, max_num_levels_to_serialize, 0);

        // The master thinks of how many subtables to exchange.
        int num_subtables_to_exchange;
        if(world.rank() == 0) {
          num_subtables_to_exchange = core::math::RandInt(2, 10);
          printf("Exchanging %d subtables.\n", num_subtables_to_exchange);
        }
        boost::mpi::broadcast(world, num_subtables_to_exchange, 0);

        // Prepare the list of subtables.
        std::vector< SubTableListType > send_subtables(world.size());
        for(int j = 0; j < world.size(); j++) {
          if(j != world.rank()) {
            for(int i = 0; i < num_subtables_to_exchange; i++) {

              // Pick a random node.
              int random_node_id =
                core::math::RandInt(
                  0, static_cast<int>(local_list_of_nodes.size()));
              int begin = local_list_of_nodes[random_node_id]->begin();
              int count = local_list_of_nodes[random_node_id]->count();
              printf("Process %d chose %d %d to send to Process %d\n",
                     world.rank(), begin, count, j);
              send_subtables[j].push_back(
                distributed_table.local_table(),
                distributed_table.local_table()->get_tree()->
                FindByBeginCount(begin, count), max_num_levels_to_serialize,
                false);
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
                0, max_num_levels_to_serialize, false);
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
              j < received_subtables_in_this_round[i].size(); j++) {
            CheckIntegrity_(
              received_subtables_in_this_round[i][j].table()->get_tree());
          }
        }

        printf("%d-th exchange done!\n\n", k);
        world.barrier();

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
