/**
 * @file allnn_main.cc
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
 * @see allnn.h
 */

// To begin, note the "@file" at the top of the previous comment
// block.  This tells Doxygen to parse the file; accordingly, a
// similar line should appear at the top of all of your files.

// Reusable code should be linked through a librule's header file.
#include "allnn.h"

int main(int argc, char* argv[]) {

  // Always initialize FASTexec with main's inputs at the beggining of
  // your program.  This reads the command line, among other things.
  fx_init(argc, argv);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(NULL, "r");

  // FASTlib classes only poison data in their default constructors;
  // declarations must be followed by Init or an equivalent function.
  Matrix references;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);


  Matrix weights_matrix;
  data::Load(weights_file_name, &weights_matrix);
  Vector weights_vector;
  weights_matrix.MakeColumnVector(0, &weights_vector);

  ////////// DUAL-TREE ALLNN /////////////////////////////////////////

  KernelVectorMult kernel_vector_mult;

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "allnn_module" under the
  // rood directory (NULL) for the AllNN object to work inside.  Here,
  // we initialize it with all parameters defined "--allnn/...=...".
  struct datanode* kernel_vector_mult_module =
      fx_submodule(NULL, "kernel_vector_mult", "kernel_vector_mult_module");

  // The Init function readies our object for action.  In this
  // example, we built the AllNN class to expect all of its inputs at
  // the time of Init; other designs might do things differently.
  kernel_vector_mult.Init(references, weights_vector, kernel_vector_mult_module);

  // ArrayLists are prefered over standard C/C++ arrays.  These grow
  // dynamically and check bounds when compiled with "--mode=debug".
  // An index_t is an integer with compile-time specified size.
  Vector results;

  // Tell the KernelVectorMult object to perform its computation. 
  kernel_vector_mult.ComputeKernelMatrixVectorMultiplication(&results);

  ////////// OUTPUT RESULTS //////////////////////////////////////////

  const char* output_filename =
      fx_param_str(NULL, "output_filename", "output.txt");

  // We encourage you to use C-style file streams and print buffers
  // rather than C++'s complicated equivalents.
  FILE* output_file = fopen(output_filename, "w");

  // The ot namespace stands for object traversal and provides many
  // features.  Here, we pretty-print an ArrayList to file, though we
  // could alternately have serialized it for later loading and reuse.
  //
  // Note that this doesn't write a .csv, but instead a transcript of
  // the ArrayList's contents.  The library does not appear to have a
  // .csv writer for ArrayLists--this will be addressed.  You can
  // write .csv from Matrix objects with data::save.
  ot::Print(results, "results", output_file);

  // Don't forget these things!
  fclose(output_file);

  // We must tell FASTexec to wrap up when our code is done.  This
  // emits its complete data structure--parameter settings, timers,
  // and stored results--to stdout unless you call fx_silence first.
  fx_done();

  // main should return 0 if the program terminates normally.
  return 0;

} /* main */
