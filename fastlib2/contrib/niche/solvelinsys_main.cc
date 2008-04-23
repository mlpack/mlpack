#include "solvelinsys.h"


int main(int argc, char *argv[]) {

  // Initialize FastExec...
  fx_init(argc, argv);


  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(NULL, "r");

  // The reference data file is a required parameter.
  const char* right_hand_side_file_name = fx_param_str_req(NULL, "rhs");

  Matrix references;
  data::Load(references_file_name, &references);

  Matrix right_hand_side_mat;
  data::Load(right_hand_side_file_name, &right_hand_side_mat);

  Vector right_hand_side_vec;
  right_hand_side_mat.MakeColumnVector(0, &right_hand_side_vec);

  Vector solution;

  double bandwidth = fx_param_double(NULL, "bandwidth", 1);
  double sigma_squared = fx_param_double(NULL, "sigma_squared", 1);

  SolveLinearSystem(references, right_hand_side_vec, bandwidth, sigma_squared, &solution);

  Matrix solution_mat;
  solution_mat.AliasColVector(solution);

  data::Save("solution.txt", solution);

  //solution.PrintDebug("solution");

  // Finalize FastExec and print output results.
  fx_done();
  return 0;
}

