/**
 * @file kernel_vector_mult_main.cc
 *
 * This file contains a "platonic" example of FASTlib code for a
 * stand-alone executable.  It makes use of an accompanying library
 * implimenting a rudimentary dual-tree all-nearest-neighbors
 * algorithm, but more importantly, it demonstrates useful functions
 * for common tasks as well as proper coding style.
 *
 * Note however that the degree of documentation in this file well
 * exceeds expectations.  You should always provide Doxygen-parsed
 * comments (those starting with slash-star-star) for classes, their
 * members, and functions, but snippets of code only deserve
 * documentation if it is not immediately clear what they do.  Here,
 * we assume you are a total beginner with FASTlib and a novice with
 * C++, so many additional explanations have been provided.  We will
 * denote explanatory comments not needed in normal coding with
 * slash-slash and the appropriate degree of code documentation with
 * slash-star and slash-star-star.
 *
 * @see kernel_vector_mult.h
 */

// To begin, note the "@file" at the top of the previous comment
// block.  This tells Doxygen to parse the file; accordingly, a
// similar line should appear at the top of all of your files.

// Reusable code should be linked through a librule's header file.
#include "kernel_vector_mult.h"

int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(NULL, "r");

  Matrix references;
  data::Load(references_file_name, &references);


  const char* weights_file_name = fx_param_str_req(NULL, "weights");
  Matrix weights_matrix;
  data::Load(weights_file_name, &weights_matrix);
  Vector weights_vector;
  weights_matrix.MakeColumnVector(0, &weights_vector);


  KernelVectorMult kernel_vector_mult;

  struct datanode* kernel_vector_mult_module =
      fx_submodule(NULL, "kernel_vector_mult");

  kernel_vector_mult.Init(references, kernel_vector_mult_module);
  
  Vector results;
  results.Init(references.n_cols());

  // Tell the KernelVectorMult object to perform its computation. 
  kernel_vector_mult.ComputeKernelMatrixVectorMultiplication(weights_vector, &results);
  const char* output_filename =
    fx_param_str(NULL, "output_filename", "output.txt");
  FILE* output_file = fopen(output_filename, "w");
  ot::Print(results, "results", output_file);
  fclose(output_file);

  Vector results2;
  results2.Init(references.n_cols());
  // Tell the KernelVectorMult object to perform its computation. 
  kernel_vector_mult.Reset();
  kernel_vector_mult.ComputeKernelMatrixVectorMultiplication(weights_vector, &results2);
  const char* output_filename2 =
    fx_param_str(NULL, "output_filename2", "output2.txt");
  FILE* output_file2 = fopen(output_filename2, "w");
  ot::Print(results2, "results", output_file2);
  fclose(output_file2);


  fx_done(NULL);

  return 0;

} /* main */
