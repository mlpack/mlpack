#ifndef KERNEL_KMEANS_H
#define KERNEL_KMEANS_H

#include "fastlib/fastlib.h"

/*
Given a kernel matrix of distances, a weights vector for the points, and the number of clusters k, do kernel kmeans
*/

class KernelKmeans {

 private:
  Matrix* kernel_matrix_;
  Vector* weights_;
  int k_;
  int n_points_;

  void InitClusters_(int** &clusters, int* &cluster_sizes) {
    clusters = (int**) malloc(sizeof(int*) * k_);

    for(int c = 0; c < k_; c++) {
      clusters[c] = (int*) malloc(sizeof(int) * n_points_);
    }

    cluster_sizes = (int*) malloc(sizeof(int) * k_);
    for(int c = 0; c < k_; c++) {
      cluster_sizes[c] = 0;
    }
  }

  
 public:

  void Init(Matrix *kernel_matrix_in,
	    Vector *weights_in,
	    int k_in) {
    kernel_matrix_ = kernel_matrix_in;
    weights_ = weights_in;
    k_ = k_in;
    n_points_ = weights_ -> length();
  }


  void Compute(int cluster_memberships[]) {

    int** current_clusters;
    int* current_cluster_sizes;
    InitClusters_(current_clusters, current_cluster_sizes);
    
    int** old_clusters;
    int* old_cluster_sizes;
    InitClusters_(old_clusters, old_cluster_sizes);
    
    for(int i = 0; i < n_points_; i++) {
      int assigned_cluster_index = rand() % k_;
      current_clusters[assigned_cluster_index][current_cluster_sizes[assigned_cluster_index]] = i;
      current_cluster_sizes[assigned_cluster_index]++;
    }

    PrintClusters("Initial Cluster Assignments",
		  current_clusters, current_cluster_sizes);
    printf("\n");



    int** temp_clusters;
    int* temp_cluster_sizes;



    double sum_weights[k_];
    char info[100];
    int max_iterations = 100;
    
    for(int iteration = 0; iteration < max_iterations; iteration++) {
    
      temp_clusters = current_clusters;
      current_clusters = old_clusters;
      old_clusters = temp_clusters;

      temp_cluster_sizes = current_cluster_sizes;
      current_cluster_sizes = old_cluster_sizes;
      old_cluster_sizes = temp_cluster_sizes;

      for(int c = 0; c < k_; c++) {
	current_cluster_sizes[c] = 0;
      }


      for(int c = 0; c < k_; c++) {
	int n_points_in_cluster = old_cluster_sizes[c];
	double sum = 0;
	for(int j = 0; j < n_points_in_cluster; j++) {
	  sum += (*weights_)[old_clusters[c][j]];
	}
	sum_weights[c] = sum;
      }



      for(int a = 0; a < n_points_; a++) {
	int min_dist_cluster = -1;
	double min_dist = DBL_MAX;

	for(int c = 0; c < k_; c++) {
	  int n_points_in_cluster = old_cluster_sizes[c];
	  int b, b_i, b_j;

	  double dist_c = kernel_matrix_ -> get(a, a);
	
	  double sum = 0;
	  for(int j = 0; j < n_points_in_cluster; j++) {
	    b = old_clusters[c][j];
	    sum += (*weights_)[b] * kernel_matrix_ -> get(b, a);
	  }
	  dist_c -= 2 * sum / sum_weights[c];

	  sum = 0;
	  for(int i = 0; i < n_points_in_cluster; i++) {
	    for(int j = 0; j < n_points_in_cluster; j++) {
	      b_i = old_clusters[c][i];
	      b_j = old_clusters[c][j];
	      sum +=
		(*weights_)[b_i] * (*weights_)[b_j] * kernel_matrix_ -> get(b_j, b_i);
	    }
	  }
	  dist_c += (sum / (sum_weights[c] * sum_weights[c]));

	  if(dist_c < min_dist) {
	    min_dist = dist_c;
	    min_dist_cluster = c;
	  }
	}
	
	
	current_clusters[min_dist_cluster][current_cluster_sizes[min_dist_cluster]] = a;
	current_cluster_sizes[min_dist_cluster]++;
	
	
      }

      sprintf(info, "Iteration %d", iteration);
      PrintClusters(info,
		    current_clusters, current_cluster_sizes);
    } // end k-means iteration


    for(int c = 0; c < k_; c++) {
      for(int j = 0; j < current_cluster_sizes[c]; j++) {
	cluster_memberships[current_clusters[c][j]] = c;
      }
    }



    for(int i = 0; i < k_; i++) {
      free(current_clusters[i]);
      free(old_clusters[i]);
    }

    free(current_clusters);
    free(old_clusters);
    free(current_cluster_sizes);
    free(old_cluster_sizes);
  }
  
  void PrintClusters(const char* info, int** clusters, int* cluster_sizes) {
    printf("---- %s ----- \n", info);
    for(int c = 0; c < k_; c++) {
      printf("\ncluster %d: ", c);
      for(int j = 0; j < cluster_sizes[c]; j++) {
	printf("%d, ", clusters[c][j]);
      }
    }
    printf("\n");
  }




  const Matrix kernel_matrix () const {
    return *kernel_matrix_;
  }

  const Vector weights () const {
    return *weights_;
  }

  int k () const {
    return k_;
  }

};





#endif /* KERNEL_KMEANS_H */
