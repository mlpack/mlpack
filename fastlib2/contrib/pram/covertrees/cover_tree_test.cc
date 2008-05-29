/**
 * @file cover_tree_test.cc
 *
 * This file test runs the AllKNN code using cover trees
 * and the dual-tree algorithm
 *
 */

#include <fastlib/fastlib.h>
#include "allknn_dfs.h"
#include <time.h>

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
void compare_neighbors(ArrayList<index_t>*, 
		       ArrayList<index_t>*);

/**
 * This is the main function which creates an object 
 * of the class AllKNN and does Recursive-breadth-first, 
 * Depth-first and Brute nearest neighbor computation
 * for a given pair of query set and reference set
 */
int main(int argc, char *argv[]) {

  fx_init(argc, argv, NULL);

  clock_t start, end, build=0, find = 0, find1 = 0, find2 = 0;
  const char *rfile = fx_param_str(NULL, "r", "my_ref_data.data");
  const char *qfile = fx_param_str(NULL, "q", "my_qry_data.data");

  Matrix refs, qrs;
  data::Load(rfile, &refs);
  data::Load(qfile, &qrs);

  // here we are just typecasting the data to the type
  // float because we want to use floats
  fx_timer_start(NULL, "data_conversion");
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
  fx_timer_stop(NULL,"data_coversion");

  //AllKNN<float> allknn;
  AllKNN<float> allknn;
  ArrayList<float> rbfs_dist, dfs_dist, brute_dist;
  ArrayList<index_t> rbfs_ind, dfs_ind, brute_ind;

  datanode *allknn_module = fx_submodule(NULL, "allknn");
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
  fx_timer_start(allknn_module, "rbfs");
  start = clock();
  allknn.RecursiveBreadthFirstSearch(&rbfs_ind, &rbfs_dist);
  end = clock();
  fx_timer_stop(allknn_module, "rbfs");
  find = end - start;
  
  // This does the depth first search of the 
  // nearest neighbors
  fx_timer_start(allknn_module, "dfs");
  start = clock();
  allknn.DepthFirstSearch(&dfs_ind, &dfs_dist);
  end = clock();
  fx_timer_stop(allknn_module, "dfs");
  find1 = end - start;

  // This does the brute computation of the 
  // nearest neighbors
  fx_timer_start(allknn_module, "brute");
  start = clock();
  allknn.BruteNeighbors(&brute_ind, &brute_dist);
  end = clock();
  fx_timer_stop(allknn_module, "brute");
  find2 = end - start;
  
  NOTIFY("Phase 1 complete");

//   NOTIFY("RBFS results");
//   print_results<float>(qsize, knn, &rbfs_ind, &rbfs_dist);

//   NOTIFY("DFS results");
//   print_results<float>(qsize, knn, &dfs_ind, &dfs_dist);

//   NOTIFY("BRUTE results");
//   print_results<float>(qsize, knn, &brute_ind, &brute_dist);
  
  float b = (float)build / (float)CLOCKS_PER_SEC;
  float f = (float)find / (float)CLOCKS_PER_SEC;
  printf("build = %f, rbfs = %f\n", b, f);
  
  float f2 = (float)find2 / (float)CLOCKS_PER_SEC;
  float f1 = (float)find1 / (float)CLOCKS_PER_SEC;
  printf("dfs = %f, brute = %f\n", f1, f2);
 
  //compare_neighbors(&rbfs_ind, &brute_ind);
  //compare_neighbors(&dfs_ind, &brute_ind); 

  //fx_param_bool(NULL, "fx/silent", 1);
  fx_done(NULL);

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

void compare_neighbors(ArrayList<index_t> *a, 
		       ArrayList<index_t> *b) {

  DEBUG_SAME_SIZE(a->size(), b->size());
  index_t *x = a->begin();
  index_t *y = a->end();
  index_t *z = b->begin();

  for(; x != y;) {
    DEBUG_ASSERT_MSG(*x++ == *z++, "neighbors are not same");
  }
  NOTIFY("Checked and passed!!");
}


/*
mnist_1_small X 2 = 6.1 + 16.74
mnist_1_small X 2 5knn = 6.14 + 43.61
mnist_1_small mnist_1_large = 92.29 + 238.86995
*/
