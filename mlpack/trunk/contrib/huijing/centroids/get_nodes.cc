#include "fastlib/fastlib.h"
#include "mlpack/allknn/allknn.h"
#include <string>

typedef AllkNN::QueryTree Tree;

void GetCenters(Tree *node, ArrayList<Vector> *centers) { 
  if (!node->is_leaf()) {
	  GetCenters(node->left(), centers);
		GetCenters(node->right(), centers);
	} else {
	  node->bound().CalculateMidpoint(centers->AddBack());	
	}
}

int main(int argc, char *argv[]) {
	FILE *fp;
	fx_init(argc, argv);
	std::string data_file = fx_param_str_req(NULL, "file");
  size_t leaf_size=fx_param_int(NULL, "lsize", 20);
	size_t knns = fx_param_int(NULL, "knns", 5);
  AllkNN allknn;
	Matrix data_for_tree;
  data::Load(data_file.c_str(), &data_for_tree);
	allknn.Init(data_for_tree, data_for_tree, leaf_size, knns);
  ArrayList<size_t> resulting_neighbors_tree;
  ArrayList<double> distances_tree;
  allknn.ComputeNeighbors(&resulting_neighbors_tree,
                              &distances_tree);
	fp=fopen("neighbors.txt", "w");
	if (fp==NULL) {
	  FATAL("Could not open results.txt for writing\n");
	}
  for(size_t i=0; i<resulting_neighbors_tree.size()/knns; i++) {
	  for(size_t j=0; j<knns; j++) {
	    fprintf(fp, "%li %li  %lg\n", i, resulting_neighbors_tree[i*knns+j],
						                              distances_tree[i*knns+j]);
		}
	}
	fclose(fp);
	ArrayList<Vector> centers;
	centers.Init();
	GetCenters(allknn.get_query_tree(), &centers);
	fp=fopen("centroind.txt", "w");
	if (fp==NULL) {
	  FATAL("Could not open centroids.txt for writing\n");
	}
	for(size_t i=0; i<centers.size(); i++) {
		for(size_t j=0; j<centers[i].length(); j++) {
	    fprintf(fp, "%lg ", centers[i][j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	fx_done();
}
 

