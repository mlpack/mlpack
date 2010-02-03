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
  mmapmm::MemoryManager<false>::allocator_->set_capacity(4294967296);
  std::string memfile=fx_param_str(fx_root, "memfile", "/scratch/gtg739c/temp_mem");
  mmapmm::MemoryManager<false>::allocator_->set_pool_name(memfile.c_str());
  mmapmm::MemoryManager<false>::allocator_->Init(fx_root);
  DiskAllNN disk_allnn;
  Matrix data_for_tree;
  std::string filename=fx_param_str_req(fx_root, "file");
  NOTIFY("Loading file...");
  data::LargeLoad(filename.c_str(), &data_for_tree);
  NOTIFY("File loaded...");
  disk_allnn.Init(data_for_tree, fx_root);
  GenVector<index_t> resulting_neighbors_tree;
  GenVector<double> resulting_distances_tree;
  NOTIFY("Computing Neighbors...");
  disk_allnn.ComputeNeighbors(&resulting_neighbors_tree, &resulting_distances_tree);
  NOTIFY("Neighbors Computed...");
  fx_done(fx_root);
}
