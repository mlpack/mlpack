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

  // The query data file defaults to the references.
  const char* queries_file_name =
      fx_param_str(NULL, "q", references_file_name);

  // FASTlib classes only poison data in their default constructors;
  // declarations must be followed by Init or an equivalent function.
  Matrix references;
  Matrix queries;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);
  data::Load(queries_file_name, &queries);

  ////////// DUAL-TREE ALLNN /////////////////////////////////////////

  AllNN allnn;

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "allnn_module" under the
  // rood directory (NULL) for the AllNN object to work inside.  Here,
  // we initialize it with all parameters defined "--allnn/...=...".
  struct datanode* allnn_module =
      fx_submodule(NULL, "allnn", "allnn_module");

  // The Init function readies our object for action.  In this
  // example, we built the AllNN class to expect all of its inputs at
  // the time of Init; other designs might do things differently.
  allnn.Init(queries, references, allnn_module);

  // ArrayLists are prefered over standard C/C++ arrays.  These grow
  // dynamically and check bounds when compiled with "--mode=debug".
  // An index_t is an integer with compile-time specified size.
  ArrayList<index_t> results;

  // Tell the AllNN object to perform its computation.  We pass it a
  // fresh ArrayList by pointer (hence the &) as a mental note that
  // this input will be modified, i.e. filled with results.
  allnn.ComputeNeighbors(&results);

  ////////// NAIVE ALLNN /////////////////////////////////////////////

  /* Compare results with naive if run with "--do_naive=true" */
  if (fx_param_bool(NULL, "do_naive", 0)) {

    // Our design of the AllNN class renders it usable only once;
    // different code could clean out intermediate results, but for
    // this example, we'll just create another instance.
    AllNN naive_allnn;

    // This time we'll use a different Init function so we can avoid
    // building the tree.  Also, note that submodules and parameters
    // may be created/accessed in-line.  (More on this shortly.)
    naive_allnn.InitNaive(queries, references,
        fx_submodule(NULL, "naive", "naive_module"));

    ArrayList<index_t> naive_results;
    naive_allnn.ComputeNaive(&naive_results);

    /* Perform a quick sanity check now that we have naive results */

    // We don't want to run the for-loop unless debugging, hence the
    // #ifdef.  For debug-only one-liners, use DEBUG_ONLY(expr) or the
    // other debugging macros.  See base/debug.h for more details.
#ifdef DEBUG
    for (index_t i = 0; i < results.size(); ++i) {
      // Prints a message if results are different.  Note the peculiar
      // syntax "%"LI"d" used to properly format index_t values; this
      // alerts printf that index_t is long if compiled as such.
      DEBUG_WARN_MSG_IF(results[i] != naive_results[i],
          "i = %"LI"d, results[i] = %"LI"d, naive_results[i] = %"LI"d",
          i, results[i], naive_results[i]);
    }
#endif /* DEBUG */

  } /* if do_naive */

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
  ot::Print(results, output_file);

  // Don't forget these things!
  fclose(output_file);

  // We must tell FASTexec to wrap up when our code is done.  This
  // emits its complete data structure--parameter settings, timers,
  // and stored results--to stdout unless you call fx_silence first.
  fx_done();

  // main should return 0 if the program terminates normally.
  return 0;

} /* main */
