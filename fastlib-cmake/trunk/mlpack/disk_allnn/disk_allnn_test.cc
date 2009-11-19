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
/*
 * =====================================================================================
 *
 *       Filename:  disk_allnn_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/11/2008 03:12:57 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

/**
 * @disk_allknn_test.cc
 * Test file for DiskAllkNN class
 */


#include "disk_allnn.h"
#include "fastlib/base/test.h"

class TestDiskAllNN {
 public:
  TestDiskAllNN(fx_module *module) {
    module_=module;
  }
  void Init() {
    mmapmm::MemoryManager<false>::allocator_= new mmapmm::MemoryManager<false>();
    mmapmm::MemoryManager<false>::allocator_->Init();
    disk_allnn_ = new DiskAllNN();
    naive_  = new DiskAllNN();
    data_for_tree_ = new Matrix();
    data_for_naive_= new Matrix();
    data::Load("test_data_3_1000.csv", data_for_tree_);
    data::Load("test_data_3_1000.csv", data_for_naive_);
 }

  void Destruct() {
    delete data_for_tree_;
    delete data_for_naive_;
    delete disk_allnn_; 
    delete naive_;
    mmapmm::MemoryManager<false>::allocator_->Destruct();
  }

  void TestTreeVsNaive1() {
    Init();
    disk_allnn_->Init(*data_for_tree_, module_);
    naive_->InitNaive(*data_for_naive_, module_);
 
    GenVector<index_t> resulting_neighbors_tree;
    GenVector<double> resulting_distances_tree;
    disk_allnn_->ComputeNeighbors(&resulting_neighbors_tree, &resulting_distances_tree);
    GenVector<index_t> resulting_neighbors_naive;
    GenVector<double> resulting_distances_naive;
    naive_->ComputeNaive(&resulting_neighbors_naive, &resulting_distances_naive);
    for(index_t i=0; i<resulting_neighbors_tree.length(); i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(resulting_distances_tree[i], resulting_distances_naive[i], 1e-5);
    }
    NOTIFY("DiskAllknn test 1 passed");
    Destruct();
  }
 
  void TestTreeVsNaive2() {
    Init();
    fx_set_param_str(module_, "splits", "median");
    disk_allnn_->Init(*data_for_tree_, module_);
    naive_->InitNaive(*data_for_naive_, module_);
 
    GenVector<index_t> resulting_neighbors_tree;
    GenVector<double> resulting_distances_tree;
    disk_allnn_->ComputeNeighbors(&resulting_neighbors_tree, &resulting_distances_tree);
    GenVector<index_t> resulting_neighbors_naive;
    GenVector<double> resulting_distances_naive;
    naive_->ComputeNaive(&resulting_neighbors_naive, &resulting_distances_naive);
    for(index_t i=0; i<resulting_neighbors_tree.length(); i++) {
      TEST_ASSERT(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
      TEST_DOUBLE_APPROX(resulting_distances_tree[i], resulting_distances_naive[i], 1e-5);
    }
    NOTIFY("DiskAllknn test 2 passed");
    Destruct();
  }
 

  void TestAll() {
    TestTreeVsNaive1();
    TestTreeVsNaive2();
 }
 
 private:
  DiskAllNN *disk_allnn_;
  DiskAllNN *naive_;
  Matrix *data_for_tree_;
  Matrix *data_for_naive_;
  fx_module *module_;
};

int main(int argc, char *argv[]) {
 fx_module *fx_root=fx_init(argc, argv, NULL); 
 fx_set_param_int(fx_root, "leaf_size", 20);
 TestDiskAllNN test(fx_root);
 test.TestAll();
}
