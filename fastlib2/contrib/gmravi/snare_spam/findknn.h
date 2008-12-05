#include <fastlib/fastlib.h>
#include "mlpack/allknn/allknn.h"
class FindAllkNN{
 public:
  AllkNN all_knn;
  Matrix queries;
  Matrix references;
    
  void Init(){
    printf("Will init the class\n");

    const char *ref_file_name="aggr_all_features_scaled_train1_pos.data";
    const char *query_file_name="aggr_all_features_scaled_train1_pos.data";

    //Read the query fike
    data::Load(query_file_name,&queries);

    //Read the references file
    data::Load(ref_file_name,&references);

    index_t k=11;
    all_knn.Init(queries,references,50,k); //leaf size is 50. Need 10 nearest neighbors
    
    printf("allknn initialized...\n");
    //Prepare datastructures to hold results

    ArrayList<index_t> resulting_neighbors_tree;
    ArrayList<double> distances;

    all_knn.ComputeNeighbors(&resulting_neighbors_tree,&distances);
    printf("The length is %d\n",resulting_neighbors_tree.size());
    printf("The closest distance is %f\n",distances[0]);

    printf("Lets print the distances..\n");
  
    //Now lets average out over the 10th nearest enighbours of all the points

    index_t total_length=resulting_neighbors_tree.size();
    double total_sum=0;
    for(index_t i=0;i<total_length/k;i++){
      total_sum=total_sum+sqrt(distances[k*(i+1)-1]);
    }

    index_t num_queries=queries.n_cols();
    printf("The average distance is %f\n",total_sum/num_queries);
  
  }
};
