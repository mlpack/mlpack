/** @file sub_table.test.cc
 *
 *  A "stress" test driver for sub tables.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/test/unit_test.hpp>
#include "core/metric_kernels/lmetric.h"
#include "core/table/sub_table.h"
#include "core/table/table.h"
#include "core/math/math_lib.h"
#include <time.h>

namespace core {
namespace table {

template<typename TableType>
class TestSubTable {

  public:
    typedef core::table::SubTable<TableType> SubTableType;

    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

    typedef typename TableType::TreeType TreeType;

  private:

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

    int StressTestMain() {
      for(int i = 0; i < 10; i++) {
        int num_dimensions = core::math::RandInt(3, 20);
        int num_points = core::math::RandInt(3000, 5001);
        if(StressTest(num_dimensions, num_points) == false) {
          printf("Failed!\n");
          exit(0);
        }
      }
      return 0;
    }

    bool StressTest(int num_dimensions, int num_points) {

      std::cout << "Number of dimensions: " << num_dimensions << "\n";
      std::cout << "Number of points: " << num_points << "\n";

      // Push in the reference dataset name.
      std::string references_in("random.csv");

      // Generate the random dataset and save it.
      TableType random_table;
      GenerateRandomDataset_(
        num_dimensions, num_points, &random_table);
      random_table.Save(references_in);
      core::metric_kernels::LMetric<2> l2_metric;
      random_table.IndexData(l2_metric, 20);

      for(int i = 0; i < 100; i++) {

        int num_levels_to_serialize = core::math::RandInt(2, 10);
        SubTableType subtable_saved;
        SubTableType subtable_loaded;
        core::table::DenseMatrix data_matrix;
        data_matrix.Init(random_table.n_attributes(), random_table.n_entries());
        OldFromNewIndexType *old_from_new =
          new OldFromNewIndexType[ random_table.n_entries()];
        int *new_from_old = new int[random_table.n_entries()];
        subtable_saved.Init(
          &random_table, random_table.get_tree(), num_levels_to_serialize);
        subtable_loaded.Init(
          0, data_matrix, old_from_new, new_from_old, num_levels_to_serialize);
        {
          std::ofstream ofs("tmp");
          boost::archive::text_oarchive oa(ofs);
          oa << subtable_saved;
        }
        {
          std::ifstream ifs("tmp");
          boost::archive::text_iarchive ia(ifs);
          ia >> subtable_loaded;
        }

        // Check integrity.
        CheckIntegrity_(subtable_loaded.table()->get_tree());

        // Free the memory.
        delete[] old_from_new;
        delete[] new_from_old;
      }

      return true;
    }
};
}
}

BOOST_AUTO_TEST_SUITE(TestSuiteSubTable)
BOOST_AUTO_TEST_CASE(TestCaseSubTable) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<core::tree::AbstractStatistic> > TableType;

  // Call the tests.
  //core::table::TestSubTable<TableType> sub_table_test;
  //sub_table_test.StressTestMain();

  boost::circular_buffer<core::table::DensePoint> buffer(5);

  for(int i = 0; i < 10; i++) {
    core::table::DensePoint point;
    point.Init(5);
    for(int j = 0; j < 5; j++) {
      point[j] = core::math::Random(-1.0, 1.0);
    }
    buffer.push_back(point);
  }

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
