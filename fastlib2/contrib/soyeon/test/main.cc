#include "fastlib/fastlib.h"
#include "lin_algebra_class.h"

int main(int argc, char *argv[]) {

  fx_init(argc,argv);

  QuadraticAlgebraClass my_objective;

  const char* current_x_file_name = fx_param_str_req(NULL, "xdata");
  const char* quadratic_term_file_name = fx_param_str_req(NULL, "qdata");
  const char* linear_term_file_name = fx_param_str_req(NULL, "ldata");

  Vector current_x;
  Matrix quad;
  Vector lin;

  data::Load(current_x_file_name, &current_x);  
  data::Load(quadratic_term_file_name, &quad);
  data::Load(linear_term_file_name, &lin);

  my_objective.Init(quad, lin);
  my_objective.ComputeObjective()


}
