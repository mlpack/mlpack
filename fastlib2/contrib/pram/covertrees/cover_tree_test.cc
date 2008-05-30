/**
 * @file cover_tree_test.cc
 *
 * This file test runs the AllKNN code using cover trees
 * and the dual-tree algorithm
 *
 */

#include <fastlib/fastlib.h>
#include "allknn.h"
#include <time.h>

const fx_entry_doc cover_tree_main_entries[] = {
  {"r", FX_REQUIRED, FX_STR, NULL,
   " A file containing the reference set.\n"},
  {"q", FX_PARAM, FX_STR, NULL,
   " A file containing the query set (defaults to the reference set).\n"},
  {"data_conversion", FX_TIMER, FX_CUSTOM, NULL,
   " A timer that stores the time required to convert the data to float.\n"},
  {"print_results", FX_PARAM, FX_BOOL, NULL,
   " A variable that decides whether we print the results or not.\n"},
  {"donaive", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do the naive computation(defaults to false).\n"},
  {"dorbfs", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do the recursive breadth first computation(defaults to true).\n"},
  {"dodfs", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do the depth first computation(defaults to false).\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc cover_tree_main_submodules[] = {
  {"allknn", &allknn_doc,
   " Responsible for doing nearest neighbor search using cover trees.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc cover_tree_main_doc = {
  cover_tree_main_entries, cover_tree_main_submodules,
  "This is a program to test run the dual tree nearest neighbors using cover trees.\n"
  "It performs the recursive breadth first, the depth first and the naive computation.\n"
};

/**
 * This functions prints the query points alongwith 
 * their k-NN and distance to them
 */
template<typename T>
void print_results(index_t, index_t, 
		   ArrayList<index_t>*, 
		   ArrayList<T>*);

/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */
void compare_neighbors(ArrayList<index_t>*, ArrayList<float>*, 
		       ArrayList<index_t>*, ArrayList<float>*);

/**
 * This is the main function which creates an object 
 * of the class AllKNN and does Recursive-breadth-first, 
 * Depth-first and Brute nearest neighbor computation
 * for a given pair of query set and reference set
 */
int main(int argc, char *argv[]) {

  fx_module *root = 
    fx_init(argc, argv, &cover_tree_main_doc);

  clock_t start, end, build=0, find = 0, find1 = 0, find2 = 0;
  const char *rfile = fx_param_str_req(root, "r");
  const char *qfile = fx_param_str(root, "q", rfile);

  Matrix refs, qrs;
  data::Load(rfile, &refs);
  data::Load(qfile, &qrs);

  // here we are just typecasting the data to the type
  // float because we want to use floats
  fx_timer_start(root, "data_conversion");
  GenMatrix<float> queries, references;
  queries.Init(qrs.n_rows(), qrs.n_cols());
  references.Init(refs.n_rows(), refs.n_cols());

  for (index_t i = 0; i < qrs.n_cols(); i++) {
    for(index_t j = 0; j < qrs.n_rows(); j++) {
      queries.set(j, i, (float) qrs.get(j, i));
    }
  }

  for (index_t i = 0; i < refs.n_cols(); i++) {
    for(index_t j = 0; j < refs.n_rows(); j++) {
      references.set(j, i, (float) refs.get(j, i));
    }
  }
  fx_timer_stop(root,"data_conversion");

  //AllKNN<float> allknn;
  AllKNN<float> allknn;
  ArrayList<float> rbfs_dist, dfs_dist, brute_dist;
  ArrayList<index_t> rbfs_ind, dfs_ind, brute_ind;

  datanode *allknn_module = fx_submodule(root, "allknn");

  index_t knn = fx_param_int(allknn_module, "knns",1);
  index_t dim = fx_param_int(allknn_module, "dim", queries.n_rows());
  index_t rsize = fx_param_int(allknn_module, "rsize", 
			       references.n_cols());
  index_t qsize = fx_param_int(allknn_module, "qsize", 
			       queries.n_cols());
  NOTIFY("|R| = %"LI"d , |Q| = %"LI"d", rsize, qsize);
  NOTIFY("%"LI"d dimensional space", dim);

  
  // Initializing the AllKNN object
  // The query and the reference set is saved in the 
  // object and cover trees are made for each of the 
  // sets
  start = clock();
  allknn.Init(queries, references, allknn_module);
  end = clock();
  build = end - start;
    
  // This does the recursive breadth first search 
  // of nearest neighbors
  if (fx_param_bool(root, "dorbfs", 1)) {
    fx_timer_start(allknn_module, "rbfs");
    start = clock();
    allknn.RecursiveBreadthFirstSearch(&rbfs_ind, &rbfs_dist);
    end = clock();
    fx_timer_stop(allknn_module, "rbfs");
    find = end - start;
  }

  // This does the depth first search of the 
  // nearest neighbors
  if (fx_param_bool(root, "dodfs", 0)) {
    fx_timer_start(allknn_module, "dfs");
    start = clock();
    allknn.DepthFirstSearch(&dfs_ind, &dfs_dist);
    end = clock();
    fx_timer_stop(allknn_module, "dfs");
    find1 = end - start;
  }

  // This does the brute computation of the 
  // nearest neighbors
  if (fx_param_bool(root, "donaive", 0)) {
    fx_timer_start(allknn_module, "brute");
    start = clock();
    allknn.BruteNeighbors(&brute_ind, &brute_dist);
    end = clock();
    fx_timer_stop(allknn_module, "brute");
    find2 = end - start;
  }

  NOTIFY("Phase 1 complete");

  if (fx_param_bool(root, "print_results", 0)) {
    
    NOTIFY("RBFS results");
    print_results<float>(qsize, knn, &rbfs_ind, &rbfs_dist);
    
    NOTIFY("DFS results");
    print_results<float>(qsize, knn, &dfs_ind, &dfs_dist);
    
    NOTIFY("BRUTE results");
    print_results<float>(qsize, knn, &brute_ind, &brute_dist);
  }

  float b = (float)build / (float)CLOCKS_PER_SEC;
  float f = (float)find / (float)CLOCKS_PER_SEC;
  NOTIFY("build = %f, rbfs = %f", b, f);
  
  float f2 = (float)find2 / (float)CLOCKS_PER_SEC;
  float f1 = (float)find1 / (float)CLOCKS_PER_SEC;
  NOTIFY("dfs = %f, brute = %f", f1, f2);
 
  printf("RBFS\n");
  fflush(NULL);
  compare_neighbors(&rbfs_ind, &rbfs_dist, &brute_ind, &brute_dist);
  printf("DFS\n");
  fflush(NULL);
  compare_neighbors(&dfs_ind, &dfs_dist, &brute_ind, &brute_dist); 

  //fx_param_bool(root, "fx/silent", 1);
  fx_done(root);

  return 0;
}

template<typename T>
void print_results(index_t num_points, index_t knn, 
		   ArrayList<index_t> *ind, 
		   ArrayList<T> *dist) {

  DEBUG_ASSERT(num_points * knn == ind->size());
  DEBUG_ASSERT(ind->size() == dist->size());

  for (index_t i = 0; i < num_points; i++) {
    NOTIFY("%"LI"d :", i+1);
    for(index_t j = 0; j < knn; j++) {
      NOTIFY("\t%"LI"d : %lf", 
	     (*ind)[knn*i+j]+1,
	     (*dist)[i*knn+knn-1-j]);
    }
  }
}

void compare_neighbors(ArrayList<index_t> *a, ArrayList<float> *da,
		       ArrayList<index_t> *b, ArrayList<float> *db) {
  
  DEBUG_SAME_SIZE(a->size(), b->size());
  index_t *x = a->begin();
  index_t *y = a->end();
  index_t *z = b->begin();

  for(index_t i = 0; x != y; x++, z++, i++) {
    if (*x != *z) { 
      printf("point %"LI"d brute: %"LI"d:%lf fast: %"LI"d:%lf\n",
	     i, *z, (*db)[i], *x, (*da)[i]);
      fflush(NULL);
    }
  }
  NOTIFY("Checked and passed!!");
}


/*
mnist_1_small X 2 = 6.1 + 16.74
mnist_1_small X 2 5knn = 6.14 + 43.61
mnist_1_small mnist_1_large = 92.29 + 238.86995
*/
