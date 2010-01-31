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
/** @file fft_kde_main.cc
 *
 *  Driver for FFT based KDE algorithm.
 *
 *  @author Dongryeol Lee (dongryel)
 */

#include <fastlib/fastlib.h>
#include "dataset_scaler.h"
#include "fft_kde.h"
#include "naive_kde.h"

/**
 * Main function which reads parameters and determines which
 * algorithms to run.
 *
 * In order to compile this driver, do:
 * fl-build fft_kde_bin --mode=fast
 *
 * In order to run this driver for the FFT-based KDE algorithm, type
 * the following (which consists of both required and optional
 * arguments) in a single command line:
 *
 * ./fft_kde_bin --data=name_of_the_reference_dataset
 *               --query=name_of_the_query_dataset
 *               --kde/bandwidth=0.0130619
 *               --kde/scaling=range
 *               --kde/fft_kde_output=fft_kde_output.txt
 *               --kde/naive_kde_output=naive_kde_output.txt
 *               --kde/do_naive
 *               --kde/num_grid_pts_per_dim=128
 *
 * Explanations for the arguments listed with possible values: 
 *
 * 1. data (required): the name of the reference dataset
 *
 * 2. query (optional): the name of the query dataset (if missing, the
 * query dataset is assumed to be the same as the reference dataset)
 *
 * 3. kde/bandwidth (required): smoothing parameter used for KDE; this
 * has to be positive.
 *
 * 4. kde/scaling (optional): whether to prescale the dataset - range:
 * scales both the query and the reference sets to be within the unit
 * hypercube [0, 1]^D where D is the dimensionality.  - none: default
 * value; no scaling
 *
 * 5. kde/do_naive (optional): run the naive algorithm after the fast
 * algorithm.  
 *
 * 6. kde/fft_kde_output (optional): if this flag is present, the
 * approximated density estimates are output to the filename provided
 * after it.
 *
 * 7. kde/naive_kde_output (optional): if this flag is present, the
 * exact density estimates computed by the naive algorithm are output
 * to the filename provided after it. This flag is not ignored if
 * --kde/do_naive flag is not present.
 *
 * 8. kde/num_grid_pts_per_dim (optional): specifies the number of
 * grid points per each dimension (for discretizing and gridding the
 * datasets). This is the only way to do any type of error control on
 * approximated KDE values. In general, the higher the value, more
 * accurate the density estimates will be. The default value is 128.
 */
int main(int argc, char *argv[]) {

  // initialize FastExec (parameter handling stuff)
  fx_init(argc, argv, NULL);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "fft_kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined "--kde/...=...".
  struct datanode *fft_kde_module =
    fx_submodule(NULL, "kde");

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(NULL, "data");

  // The query data file defaults to the references.
  const char* queries_file_name =
    fx_param_str(NULL, "query", references_file_name);

  // query and reference datasets
  Matrix references;
  Matrix queries;

  // flag for telling whether references are equal to queries
  bool queries_equal_references =
    !strcmp(queries_file_name, references_file_name);

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);
  if(queries_equal_references) {
    queries.Alias(references);
  }
  else {
    data::Load(queries_file_name, &queries);
  }

  // confirm whether the user asked for scaling of the dataset
  if(!strcmp(fx_param_str(fft_kde_module, "scaling", "none"), "range")) {
    DatasetScaler::ScaleDataByMinMax(queries, references,
                                     queries_equal_references);
  }

  // declare FFT-based KDE computation object and the vector holding the
  // results
  FFTKde fft_kde;
  Vector fft_kde_results;

  // initialize and compute
  fft_kde.Init(queries, references, fft_kde_module);
  fft_kde.Compute();
  fft_kde.get_density_estimates(&fft_kde_results);

  // print out the results if the user specified the flag for output
  if(fx_param_exists(fft_kde_module, "fft_kde_output")) {
    fft_kde.PrintDebug();
  }
  
  // do naive computation and compare to the FFT computations if the
  // user specified --do_naive flag
  if(fx_param_exists(fft_kde_module, "do_naive")) {
    NaiveKde<GaussianKernel> naive_kde;
    naive_kde.Init(queries, references, fft_kde_module);
    naive_kde.Compute();
    
    if(fx_param_exists(fft_kde_module, "naive_kde_output")) {
      naive_kde.PrintDebug();
    }
    naive_kde.ComputeMaximumRelativeError(fft_kde_results);
  }
  
  fx_done(NULL);
  return 0;
}
