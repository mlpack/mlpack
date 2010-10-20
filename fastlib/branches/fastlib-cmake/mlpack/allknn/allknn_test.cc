/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @allknn_test.cc
 * Test file for AllkNN class
 */

#include "allknn.h"
#include "fastlib/base/test.h"
#include <boost/program_options.hpp>

using namespace std;
namespace boost_po = boost::program_options;

boost_po::variables_map vm;

class TestAllkNN {
 public:
  void Init() {
    allknn_ = new AllkNN();
    naive_  = new AllkNN();
    data_for_tree_ = new Matrix();
    data::Load("test_data_3_1000.csv", data_for_tree_);
 }

  void Destruct() {
   delete data_for_tree_;
   delete allknn_; 
   delete naive_;
  }

  void TestDualTreeVsNaive1() {
    Init();
    allknn_->Init(*data_for_tree_, *data_for_tree_, 20, 5);
    naive_->InitNaive(*data_for_tree_, *data_for_tree_, 5);
 
    ArrayList<index_t> resulting_neighbors_tree;
    ArrayList<double> distances_tree;
    allknn_->ComputeNeighbors(&resulting_neighbors_tree,
                              &distances_tree);
    ArrayList<index_t> resulting_neighbors_naive;
    ArrayList<double> distances_naive;
    naive_->ComputeNaive(&resulting_neighbors_naive,
                         &distances_naive);
    for(index_t i=0; i<resulting_neighbors_tree.size(); i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(distances_tree[i], distances_naive[i], 1e-5);
    }
    NOTIFY("Allknn test 1 passed");
    Destruct();
  }
   void TestDualTreeVsNaive2() {
    Init();
    allknn_->Init(*data_for_tree_, 20, 5);
    naive_->InitNaive(*data_for_tree_, 5);

    ArrayList<index_t> resulting_neighbors_tree;
    ArrayList<double> distances_tree;
    allknn_->ComputeNeighbors(&resulting_neighbors_tree,
                              &distances_tree);
    ArrayList<index_t> resulting_neighbors_naive;
    ArrayList<double> distances_naive;
    naive_->ComputeNaive(&resulting_neighbors_naive,
                         &distances_naive);
    for(index_t i=0; i<resulting_neighbors_tree.size(); i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(distances_tree[i], distances_naive[i], 1e-5);
    }
    NOTIFY("Allknn test 2 passed");
    Destruct();
  }
    void TestSingleTreeVsNaive() {
    Init();
    allknn_->Init(*data_for_tree_, 20, 5, "single");
    naive_->InitNaive(*data_for_tree_, 5);

    ArrayList<index_t> resulting_neighbors_tree;
    ArrayList<double> distances_tree;
    allknn_->ComputeNeighbors(&resulting_neighbors_tree,
                              &distances_tree);
    ArrayList<index_t> resulting_neighbors_naive;
    ArrayList<double> distances_naive;
    naive_->ComputeNaive(&resulting_neighbors_naive,
                         &distances_naive);
    for(index_t i=0; i<resulting_neighbors_tree.size(); i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(distances_tree[i], distances_naive[i], 1e-5);
    }
    NOTIFY("Allknn test 3 passed");
    Destruct();
  }

  void TestAll() {
    TestDualTreeVsNaive1();
    TestDualTreeVsNaive2();
    TestSingleTreeVsNaive();
 }
 
 private:
  AllkNN *allknn_;
  AllkNN *naive_;
  Matrix *data_for_tree_;
};

int main(int argc, char** argv) {

  boost_po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "Display options")
    ("reference_file", boost_po::value<std::string>(), "  The reference file name")
    ("result_file", boost_po::value<std::string>(), "  The result file name")
    ("query_file", boost_po::value<std::string>(), "Number of nearest neighbours" )
    ("leaf_size", boost_po::value<index_t>(), "The number of points in a leaf")
    ("mode", boost_po::value<std::string>() , "This can be either single or dual referring to dual tree and single tree algorithm")
    ("knns", boost_po::value<int>(), "Number of nearest neighbours" );

  boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
  boost_po::notify(vm);

  if( vm.count("help"))
  {
    cout  << desc << endl;
    exit(1);
  }
  
  fx_module *module = fx_init(argc, argv, NULL);
  TestAllkNN test;
  test.TestAll();
}
