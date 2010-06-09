/*
 *  make_matchers.cc
 *  
 *
 *  Created by William March on 6/9/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */


#include "results_tensor.h"
#include "fastlib/fastlib.h"


int main(int argc, char* argv[]) {

  int n = 2;
  int num_bands = 5;
  double band_min = 0.01;
  double band_max = 0.1;
  
  ArrayList<double> dists_sq;
  dists_sq.Init(num_bands);
  
  
  double this_dist = band_min;
  double dist_step = (band_max - band_min) / (double)(num_bands-1);
  
  //printf("dist_step = %g\n", dist_step);
  
  // TODO: double check this
  for (index_t i = 0; i < num_bands; i++) {
    
    dists_sq[i] = this_dist * this_dist;
    this_dist += dist_step;
    
  }
  
  
  ResultsTensor rt;
  rt.Init(n, num_bands);
  
  const char* output_file = "test_matchers/n_2_bins_5.txt";
  FILE* fp = fopen(output_file, "w");
  
  rt.Output(dists_sq, fp);
  
  fclose(fp);
  
  return 0;
  
} // main()