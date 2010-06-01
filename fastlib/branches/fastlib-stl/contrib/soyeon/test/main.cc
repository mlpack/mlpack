#include "fastlib/fastlib.h"
#include "lin_algebra_class.h"
#include "iostream.h"

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

  NOTIFY("The objective is %lg", dummy_objective);
  fx_done(fx_root);

	Vector dummy_gradient;
  my_objective.ComputeGradient(initial_x, &dummy_gradient);
	
	/*Matrix dummy_hessian;
	my_objective.ComputeHessian(initial_x, &dummy_hessian);
  
	cout<<"Objective function value: "<< dummy_objective << "\n";
*/

	int i,j,k;
	cout<<"Gradient vector: ";
	for (i=0; i<dummy_gradient.length(); i++)
	{
		cout<<dummy_gradient[i]<<" ";
	}
	cout<<endl;
	/*
	cout<<"Hessian matrix: ";
	for (j=0; j<dummy_hessian.n_rows(); j++)
	{
		for (k=0; k<dummy_hessian.n_rows(); k++)
		{
			cout<<dummy_hessian[j][k] <<"  ";
			//cout<<dummy_hessian.get(j,k) <<"  ";
			cout<<endl;
		}
	}
	
	data::Save("Hessian.csv", dummy_hessian);

>>>>>>> .r2779
*/

  fx_done(fx_root);





}
