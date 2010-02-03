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
 *
 * Test driver for our infomax ICA method.
 */

#include "infomax_ica.h"
#include "test_infomax_ica.h"
#include "fastlib/fastlib.h"

const fx_entry_doc infomax_ica_main_entries[] = {
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

  TestInfomaxICA *testica = new TestInfomaxICA();
  testica->Init();
  testica->TestAll();

  fx_done(NULL);
}
