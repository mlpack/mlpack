/*
 * =====================================================================================
 *
 *       Filename:  main.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/14/2008 07:15:55 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include <string>
#include "disk_allnn.h"

int main (int argc, char *argv[]) {
  fx_module *fx_root=fx_init(argc, argv, NULL);
  mmapmm::MemoryManager<false>::allocator_= new mmapmm::MemoryManager<false>();
  mmapmm::MemoryManager<false>::allocator_->set_capacity(1073741824);
 mmapmm::MemoryManager<false>::allocator_->set_pool_name("/scratch/gtg739c/temp_mem");
  mmapmm::MemoryManager<false>::allocator_->Init();
  DiskAllNN disk_allnn;
  Matrix data_for_tree;
  std::string filename=fx_param_str_req(fx_root, "file");
  NOTIFY("Loading file...");
  data::Load(filename.c_str(), &data_for_tree);
  NOTIFY("File loaded...");
  disk_allnn.Init(data_for_tree, fx_root);
  GenVector<index_t> resulting_neighbors_tree;
  GenVector<double> resulting_distances_tree;
  NOTIFY("Computing Neighbors...");
  disk_allnn.ComputeNeighbors(&resulting_neighbors_tree, &resulting_distances_tree);
  NOTIFY("Neighbors Computed...");
  fx_done(fx_root);
}
