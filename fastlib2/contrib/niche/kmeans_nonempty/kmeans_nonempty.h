#include "fastlib/fastlib.h"
#include "do_mcf.h"


void WriteProblem(const Matrix &distances_sq, int min_points_per_cluster) {
  int n_clusters = distances_sq.n_rows();
  int n_points = distances_sq.n_cols();
  
  const char* filename = "clustering_problem";
  FILE* file = fopen(filename, "w");
 

  int n_nodes = n_points + n_clusters + 1;
  int n_arcs = (n_points * n_clusters) + n_clusters;

  int cluster_ids[n_clusters];
  for(int j = 0; j < n_clusters; j++) {
    cluster_ids[j] = n_points + j + 1;
  }

  int sink_id = n_points + n_clusters + 1;
 
  fprintf(file, "p min %d %d\n", n_nodes, n_arcs);
  for(int i = 0; i < n_points; i++) {
    fprintf(file, "n %d 1\n", i + 1);
  }
  for(int j = 0; j < n_clusters; j++) {
    fprintf(file, "n %d %d\n", cluster_ids[j], -min_points_per_cluster);
  }
  fprintf(file, "n %d %d\n",
	  sink_id, -n_points + (n_clusters * min_points_per_cluster));
  for(int i = 0; i < n_points; i++) {
    for(int j = 0; j < n_clusters; j++) {
      fprintf(file, "a %d %d 0 1 %f\n",
	      i + 1, cluster_ids[j],
	      distances_sq.get(j, i));
    }
  }
  for(int j = 0; j < n_clusters; j++) {
    fprintf(file, "a %d %d 0 free 0\n",
	    cluster_ids[j], sink_id);
  }

  fclose(file);
}

// p_cluster_memberships must point to an Init'd Vector
void ReadSolution(GenVector<int>* p_cluster_memberships, int* p_n_changes) {
  GenVector<int> &cluster_memberships = *p_cluster_memberships;
  int &n_changes = *p_n_changes;
  
  int n_points = cluster_memberships.length();

  FILE* file = fopen("clustering_problem.sol", "r");
  
  char* buffer = (char*) malloc(80 * sizeof(char));
  for(int i = 0; i < 5; i++) {
    fgets(buffer, 80, file);
  }
  free(buffer);

  int source_id;
  int dest_id;
  int flow;
  int n_points_plus_1 = n_points + 1;
  n_changes = 0;
  for(int i = 0; i < n_points; i++) {
    fscanf(file, "f %d %d %d\n", &source_id, &dest_id, &flow);
    int point_num = source_id - 1;
    int cluster_num = dest_id - n_points_plus_1;
    if(cluster_memberships[point_num] != cluster_num) {
      n_changes++;
      cluster_memberships[point_num] = cluster_num;
    }
    cluster_memberships[source_id - 1] = dest_id - n_points_plus_1;
  }
  fclose(file);

  //for(int i = 0; i < n_points; i++) {
  //  printf("%d ", cluster_memberships[i]);
  //}
  //printf("\n");
}



template<typename T>
void ConstrainedKMeans(const ArrayList<GenMatrix<T> > &datasets,
		       int n_clusters,
		       int max_iterations,
		       int min_points_per_cluster,
		       const char* problem_filename,
		       const char* solution_filename,
		       GenVector<int>* p_cluster_memberships,
		       int cluster_counts[]) {
  GenVector<int> &cluster_memberships= *p_cluster_memberships;

  int n_dims = datasets[0].n_rows();
    
  ArrayList<Vector> cluster_centers;
  cluster_centers.Init(n_clusters);
  for(int k = 0; k < n_clusters; k++) {
    cluster_centers[k].Init(n_dims);
  }
    
  int n_datasets = datasets.size();
  int n_points = 0;
  for(int m = 0; m < n_datasets; m++) {
    n_points += datasets[m].n_cols();
  }

  int i;

  // random initialization
  cluster_memberships.Init(n_points);
  for(i = 0; i < n_points; i++) {
    cluster_memberships[i] = rand() % n_clusters;
    printf("cluster_memberships[%d] = %d\n", i, cluster_memberships[i]);
  }

  int iteration_num = 1;
  int n_changes = 0;
  bool converged = false;
  while(!converged) {
    // update cluster centers using cluster memberships
    for(int k = 0; k < n_clusters; k++) {
      cluster_centers[k].SetZero();
      cluster_counts[k] = 0;
    }
    i = 0;
    for(int m = 0; m < n_datasets; m++) {
      const GenMatrix<T> &data = datasets[m];
      int n_points_in_data = data.n_cols();
      for(int j = 0; j < n_points_in_data; j++) {
	Vector point;
	data.MakeColumnVector(j, &point);
	int cluster_index = cluster_memberships[i];
	la::AddTo(point, &(cluster_centers[cluster_index]));
	cluster_counts[cluster_index]++;
	i++;
      }
    }
    for(int k = 0; k < n_clusters; k++) {
      la::Scale(((double)1) / ((double)(cluster_counts[k])),
		&(cluster_centers[k]));
      //cluster_centers[k].PrintDebug("cluster");
    }
   
    // update cluster memberships using cluster centers
    Matrix distances_sq;
    distances_sq.Init(n_clusters, n_points);
    i = 0;
    for(int m = 0; m < n_datasets; m++) {
      const GenMatrix<T> &data = datasets[i];
      int n_points_in_data = data.n_cols();
      for(int j = 0; j < n_points_in_data; j++) {
	Vector point;
	data.MakeColumnVector(j, &point);
	for(int k = 0; k < n_clusters; k++) {
	  distances_sq.set(k, i, 
			   la::DistanceSqEuclidean(point,
						   cluster_centers[k]));
	}
	i++;
      }
    }
    
    WriteProblem(distances_sq, min_points_per_cluster);
    DoMCF(problem_filename, solution_filename);
    ReadSolution(&cluster_memberships, &n_changes);

    printf("n_changes = %d\n", n_changes);

    iteration_num++;
    if((n_changes == 0) || (iteration_num > max_iterations)) {
      converged = true;
    }
  }
  printf("converged in %d iterations\n", iteration_num);
}
