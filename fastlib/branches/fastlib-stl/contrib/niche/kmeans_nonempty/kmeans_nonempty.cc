#include "fastlib/fastlib.h"
#include "kmeans_nonempty.h"

int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);
  /*
  DoMCF();
  GenVector<int> cluster_memberships;
  ReadSolution(60000, &cluster_memberships);
  */

  
  int n_datasets = 1;
  int n_dims = 3;
  int n_points = 1000;
  
  
  ArrayList<GenMatrix<double> > datasets;
  datasets.Init(n_datasets);
  data::Load("data.csv", &(datasets[0]));
  datasets[0].PrintDebug("dataset");
  n_dims = datasets[0].n_rows();
  n_points = datasets[0].n_cols();
  printf("n_dims = %d\n", n_dims);
  printf("n_points = %d\n", n_points);

  /*
  for(int m = 0; m < n_datasets; m++) {
    datasets[m].Init(n_dims, n_points);
    for(int i = 0; i < n_points; i++) {
      for(int j = 0; j < n_dims; j++) {
	datasets[m].set(j, i, drand48());
      }
    }
    datasets[m].PrintDebug("dataset");

  }
  */
 
  int n_clusters = 4;
  int max_iterations = 50;
  int min_points_per_cluster = 20;

  GenVector<int> cluster_memberships;
  int cluster_counts[n_clusters];

      
  const char* problem_filename = "clustering_problem";
  const char* solution_filename = "clustering_problem.sol";
  
  ConstrainedKMeans(datasets,
		    n_clusters,
		    max_iterations,
		    min_points_per_cluster,
		    problem_filename,
		    solution_filename,
		    &cluster_memberships,
		    cluster_counts);

  /*			
  Matrix data;
  data::Load("data.csv", &data);

  Matrix distances_sq;
  data::Load("distances_sq.csv", &distances_sq);
  fx_timer_start(NULL, "write_problem");
  WriteProblem(distances_sq, 10);
  fx_timer_stop(NULL, "write_problem");
  */
  fx_done(root);
}

