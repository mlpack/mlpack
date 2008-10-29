#include "fastlib/fastlib.h"
#include "lin_algebra_class.h"

int main(int argc, char *argv[]) {

//<<<<<<< .mine
  //fx_init(argc,argv);
//=======
  //parsing
  fx_module *fx_root = fx_init(argc, argv, NULL);
//>>>>>>> .r2718

  QuadraticObjective my_objective;

/*<<<<<<< .mine
  const char* current_x_file_name = fx_param_str_req(NULL, "xdata");
  const char* quadratic_term_file_name = fx_param_str_req(NULL, "qdata");
  const char* linear_term_file_name = fx_param_str_req(NULL, "ldata");
=======
>>>>>>> .r2718*/

  //optional input file, default file "xinit.csv"
  const char* initial_solution_file = fx_param_str(fx_root, "xinit_file", "xinit.csv");
  
  //Reguired input file
  const char* quadratic_term_file = fx_param_str_req(fx_root, "qdata");
  const char* linear_term_file = fx_param_str_req(fx_root, "ldata");
  //index_t problem_id = fx_param_int_req(fx_root, "problem_id");

  Vector initial_x;
  Matrix quadratic_term;
  Vector linear_term;
  
  //fastlib only load data as Matrix
  Matrix initial_x_mat;
  if (data::Load(initial_solution_file, &initial_x_mat)==SUCCESS_FAIL) {
    FATAL("File %s not found", initial_solution_file);
  };  
  initial_x_mat.MakeColumnVector(0, &initial_x);
  if (data::Load(quadratic_term_file, &quadratic_term)==SUCCESS_FAIL) {
    FATAL("File %s not found", quadratic_term_file);
  } 
  Matrix linear_term_mat;
  data::Load(linear_term_file, &linear_term_mat);
  linear_term_mat.MakeColumnVector(0, &linear_term);

  my_objective.Init(quadratic_term, linear_term);
  double dummy_objective;
  my_objective.ComputeObjective(initial_x, &dummy_objective);
  

  fx_done(fx_root);





}
