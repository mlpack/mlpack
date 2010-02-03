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
 * @file main.cc
 * @author Chip Mappus
 *
 * main for using infomax ICA method.
 */

#include "infomax_ica.h"
#include "test_infomax_ica.h"
#include "fastlib/fastlib.h"
#include "fastlib/data/dataset.h"

const fx_entry_doc infomax_ica_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   "  The name of the file containing mixture data.\n"},
  {"lambda", FX_PARAM, FX_DOUBLE, NULL,
   "  The learning rate.\n"},
  {"B", FX_PARAM, FX_INT, NULL,
   "  Infomax data window size.\n"},
  {"epsilon", FX_PARAM, FX_DOUBLE, NULL,
   "  Infomax algorithm stop threshold.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc infomax_ica_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc infomax_ica_main_doc = {
  infomax_ica_main_entries, infomax_ica_main_submodules,
  "This performs ICA decomposition on a given dataset using the Infomax method.\n"
};

int main(int argc, char *argv[]) {
  fx_module *root = fx_init(argc, argv, &infomax_ica_main_doc);

  const char *data_file_name = fx_param_str_req(root, "data");
  double lambda = fx_param_double(root,"lambda",0.001);
  int B = fx_param_int(root,"B",5);
  double epsilon = fx_param_double(root,"epsilon",0.001);
  Matrix dataset;
  data::Load(data_file_name,&dataset);
  InfomaxICA *ica = new InfomaxICA(lambda, B, epsilon);

  ica->applyICA(dataset);  
  Matrix west;
  ica->getUnmixing(west);
  //ica->displayMatrix(west);

  fx_done(NULL);
}
