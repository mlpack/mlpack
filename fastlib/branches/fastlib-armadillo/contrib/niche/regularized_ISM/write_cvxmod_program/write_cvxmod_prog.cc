#include "fastlib/fastlib.h"
#include "mlpack/allknn/allknn.h"


int main(int argc, char* argv[]) {
  fx_module* module = fx_init(argc, argv, NULL);

  std::string data_file = fx_param_str_req(module, "data");
  std::string labels_file = fx_param_str_req(module, "labels");
  Matrix data;
  Matrix labels_mat;
  ArrayList<index_t> neighbors;
  ArrayList<double> distances;
  if (data::Load(data_file.c_str(), &data)==SUCCESS_FAIL) {
    FATAL("Data file %s not found", data_file.c_str());
  }
  NOTIFY("Loaded data from file %s", data_file.c_str());

  index_t n_points = data.n_cols();


  if (data::Load(labels_file.c_str(), &labels_mat)==SUCCESS_FAIL) {
    FATAL("Labels file %s not found", labels_file.c_str());
  }
  NOTIFY("Loaded labels from file %s", labels_file.c_str());

  Vector labels;
  labels.Init(n_points);
  labels.SetZero();

  for(index_t i = 1; i < labels_mat.n_cols(); i++) {
    labels[i-1] = labels_mat.get(0, i);
  }

 
  AllkNN allknn; 
  NOTIFY("Building reference tree");
  allknn.Init(data, module);

  NOTIFY("Tree(s) built");
  index_t knns = fx_param_int_req(module, "knns");
  NOTIFY("Computing %"LI"d nearest neighbors", knns);
  allknn.ComputeNeighbors(&neighbors, &distances);
  NOTIFY("Neighbors computed");
  NOTIFY("Exporting results");

  // Note: Distances are actually squared distances.

  index_t n_constraints = n_points * knns;

  Matrix constraints;
  constraints.Init(n_constraints, 3);
  
  int constraint_num = 0;
  for(index_t i = 0 ; i < neighbors.size()/knns ; i++) {
    for(index_t j = 0; j < knns; j++) {
      constraints.set(constraint_num, 0, i);
      constraints.set(constraint_num, 1, neighbors[i * knns + j]);
      constraints.set(constraint_num, 2, distances[i * knns + j]);
      constraint_num++;
    }
  }


  int killed_constraints = 0;
  for(index_t i = 0; i < n_constraints; i++) {
    if(labels[(int)constraints.get(i, 0)] * labels[(int)constraints.get(i, 1)] < -0.5) {
      constraints.set(i, 2, -1); // mark discarded constraints
      killed_constraints++;
    }
  }

  printf("%d constraints killed due to mismatched labels\n", killed_constraints);


  // prepare to write python code
  FILE* fid = fopen("problem_x.py", "w");

  // import modules
  fprintf(fid, "from cvxmod import *\n");
  fprintf(fid, "from cvxmod.atoms import log\n\n");

  // declare K such that K \succeq 0 and K = K^T
  fprintf(fid, "K = optvar('K', %d, %d, symm = True, psd = True)\n\n",
	  n_points + 1, n_points + 1);

  // declare bias
  //fprintf(fid, "b = optvar('b', 1)\n\n");

  // declare slack variables xi such that xi >= 0
  fprintf(fid, "xi = optvar('xi', %d, pos = True)\n\n", n_points);

  fprintf(fid, "constraints = [ ]\n");

  // centering constraint
  fprintf(fid, "constraints += [ sum(sum(K[0:-1,0:-1])) == 0 ]\n");

  // isometry constraints
  for(index_t i = 0; i < n_constraints; i++) {
    if(constraints.get(i, 2) != -1) {
      fprintf(fid, 
	      "constraints += [ K[%d,%d] + K[%d,%d] - 2 * K[%d,%d] == %f ]\n",
	      (int) constraints.get(i, 0), (int) constraints.get(i, 0),
	      (int) constraints.get(i, 1), (int) constraints.get(i, 1),
	      (int) constraints.get(i, 0), (int) constraints.get(i, 1),
	      constraints.get(i, 2));
    }
  }

  fprintf(fid, "\n");
  
  int n_unlabeled_points = 0;
  // soft margin constraints
  for(index_t i = 0; i < n_points; i++) {
    // since labels holds double, capture -1 and +1 cases via inequalities
    if((labels[i] < -0.5) || (0.5 < labels[i])) {
      fprintf(fid, 
	      "constraints += [ %f * (K[%d, -1]) + xi[%d] >= 1 ]\n",
	      labels[i], i, i);
    }
    else {
      n_unlabeled_points++;
    }
  }
  
  printf("%d unlabeled points\n", n_unlabeled_points);


  // define problem


  double lambda_s = 1;
  double lambda_k = 1;

  const char* min_margin_objective = "K[-1, -1]";
  char* min_trace_objective = (char*) malloc(100 * sizeof(char));
  sprintf(min_trace_objective, "%f * trace(K[0:-1,0:-1])", lambda_k);
  char* min_softness_objective = (char*) malloc(100 * sizeof(char));
  sprintf(min_softness_objective, "%f * sum(xi)", lambda_s);

  fprintf(fid, "\np = problem(minimize(%s + %s + %s), constraints)\n",
	  min_margin_objective,
	  min_trace_objective,
	  min_softness_objective);

  free(min_trace_objective);
  free(min_softness_objective);

  // solve problem
  fprintf(fid, "p.solve()\n");

  // finished writing python code
  fclose(fid);





  fx_done(module);
}
