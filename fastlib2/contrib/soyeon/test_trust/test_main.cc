#include "fastlib/fastlib.h"
#include "test_obj.h"
#include "optimization.h"
#include "iostream.h"


int main(int argc, char *argv[]) {


  fx_module *fx_root = fx_init(argc, argv, NULL);


  RosenbrockFunction my_test;

  //optional input file, default file "xinit.csv"
  const char* initial_solution_file = fx_param_str(fx_root, "xinit_file", "xinit.csv");
  
  /*
	//Reguired input file
  const char* quadratic_term_file = fx_param_str_req(fx_root, "qdata");
  const char* linear_term_file = fx_param_str_req(fx_root, "ldata");
  //index_t problem_id = fx_param_int_req(fx_root, "problem_id");

  Vector initial_x;
  Matrix quadratic_term;
  Vector linear_term;
  */

  //fastlib only load data as Matrix
	Vector initial_x;
  Matrix initial_x_mat;
  if (data::Load(initial_solution_file, &initial_x_mat)==SUCCESS_FAIL) {
    FATAL("File %s not found", initial_solution_file);
  };  
  initial_x_mat.MakeColumnVector(0, &initial_x);
  
	/*
	if (data::Load(quadratic_term_file, &quadratic_term)==SUCCESS_FAIL) {
    FATAL("File %s not found", quadratic_term_file);
  } 
  Matrix linear_term_mat;
  data::Load(linear_term_file, &linear_term_mat);
  linear_term_mat.MakeColumnVector(0, &linear_term);
	*/

  //my_test.Init(quadratic_term, linear_term);
  
	double dummy_objective;
  my_test.ComputeObjective(initial_x, &dummy_objective);
  NOTIFY("The objective is %lg", dummy_objective);
 
	Vector dummy_gradient;
  my_test.ComputeGradient(initial_x, &dummy_gradient);
	
	Matrix dummy_hessian;
	my_test.ComputeHessian(initial_x, &dummy_hessian);
  
	

  fx_done(fx_root);





}


  
