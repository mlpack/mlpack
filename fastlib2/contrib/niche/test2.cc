#include "fastlib/fastlib.h"
#include "fastlib/sparse/trilinos/include/AztecOO.h"
#include "fastlib/sparse/trilinos/include/Epetra_BlockMap.h"
#include "fastlib/sparse/trilinos/include/Epetra_CrsMatrix.h"
#include "fastlib/sparse/trilinos/include/Epetra_LinearProblem.h"
#include "fastlib/sparse/trilinos/include/Epetra_Map.h"
#include "fastlib/sparse/trilinos/include/Epetra_MultiVector.h"
#include "fastlib/sparse/trilinos/include/Epetra_Operator.h"
#include "fastlib/sparse/trilinos/include/Epetra_SerialComm.h"
#include "fastlib/sparse/trilinos/include/Epetra_Vector.h"
#include "contrib/dongryel/regression/multi_conjugate_gradient.h"
#include "contrib/dongryel/regression/krylov_lpr.h"
#include "contrib/dongryel/regression/relative_prune_lpr.h"
#include "fastlib/sparse/trilinos/include/Epetra_SerialDenseMatrix.h"
#include "kernel_vector_mult.h"


class SimpleLinearOperator: public virtual Epetra_Operator {
  FORBID_ACCIDENTAL_COPIES(SimpleLinearOperator);

private:
  

  int n_points_;
  int n_dims_;

  // used for a simple linear transformation operator data_^T * data  
  Matrix data_;
  //  Matrix K_;

  EpanKernel epan_kernel_;
  double norm_constant_;

  double sigma_;

  KernelVectorMult* kernel_vector_mult_;


  
  
public:
  
  SimpleLinearOperator(int n_points_in, Matrix data_in,
		       double bandwidth_in, double sigma_in,
		       KernelVectorMult* kernel_vector_mult_in) {

    n_points_ = n_points_in;
    kernel_vector_mult_ = kernel_vector_mult_in;

    
    data_.Copy(data_in);

    //la::MulTransBInit(data_in, data_in, &K_);
    //data::Load("K.txt", &K_);
    
    n_dims_ = data_in.n_rows();
    //n_points_ = data_in.n_cols();



    map = new Epetra_Map(n_points_, 0, comm);

    epan_kernel_.Init(bandwidth_in, n_dims_);
    norm_constant_ = epan_kernel_.CalcNormConstant(n_dims_);

    sigma_ = sigma_in;


  }

  ~SimpleLinearOperator() {
    delete map;
  }


  int SetUseTranspose(bool UseTranspose) {
    return false;
  }

  int Apply (const Epetra_MultiVector &X, Epetra_MultiVector &Y) const {
    
    /* fast summation code */
    Vector weights_vector;
    Vector results;
    weights_vector.Init(n_points_);
    results.Init(n_points_);


    for(int j = 0; j < n_points_; j++) {
      weights_vector[j] = X.Pointers()[0][j];
    }

    kernel_vector_mult_ -> Reset();
    kernel_vector_mult_ -> ComputeKernelMatrixVectorMultiplication(weights_vector, &results);


    for(int i = 0; i < n_points_; i++) {
      Y.Pointers()[0][i] = results[i] + (sigma_ * weights_vector[i]);
    }

    /* end fast summation code */




    /*
    // LET'S ASSUME THAT DATA IS A D X N MATRIX

    Vector K_i;
    K_i.Init(n_points_);


    for(int i = 0; i < n_points_; i++) {
      Vector v_i;
      data_.MakeColumnVector(i, &v_i);
      
      for(int j = 0; j < n_points_; j++) {
	Vector v_j;
	data_.MakeColumnVector(j, &v_j);
	
	double dist = la::DistanceSqEuclidean(v_i, v_j);
	
	K_i[j] = epan_kernel_.EvalUnnormOnSq(dist) / norm_constant_;
      }
      
      //K_i[i] += sigma_; // add (sigma * I) term

      double sum = 0;
      for(int j = 0; j < n_points_; j++) {
	sum += K_i[j] * X[0][j];
      }
      
      Y.Pointers()[0][i] = sum;
    }

    double my_squared_error = 0;
    for(int i = 0; i < n_points_; i++) {
      my_squared_error += pow(Y.Pointers()[0][i] - results[i], 2);
    }

    printf("my_squared_error = %f\n", my_squared_error);
    */
    


    return 0;
  }

  int ApplyInverse(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const {
    return -1;
  }
  
  double NormInf() const {
    return -1;
  }
  
  const char *Label() const {
    return "Simple Linear Operator";
  }

  bool UseTranspose() const {
    return false;
  }
  
  bool HasNormInf() const {
    return false;
  }
  
  const Epetra_Comm &Comm() const {
    return comm;
  }

  const Epetra_Map &OperatorDomainMap() const {
    const Epetra_Map &map_reference = *map;
    return map_reference;
  }

  const Epetra_Map &OperatorRangeMap() const {
    const Epetra_Map &map_reference = *map;
    return map_reference;
  }

public:
  
  AztecOO *solver;
  Epetra_SerialComm comm;
  Epetra_Map *map;
  
};





int main(int argc, char *argv[]) {

  // Initialize FastExec...
  fx_init(argc, argv);



  /* begin code from KernelVectorMult */
  
  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(NULL, "r");
  
  Matrix references;
  data::Load(references_file_name, &references);

  printf("references.n_rows() = %d\nreferences.n_cols() = %d\n",
	 references.n_rows(), references.n_cols());

  
  KernelVectorMult kernel_vector_mult;
  
  struct datanode* kernel_vector_mult_module =
    fx_submodule(NULL, "kernel_vector_mult", "kernel_vector_mult_module");
  
  kernel_vector_mult.Init(references, kernel_vector_mult_module);
  
  /* end code from KernelVectorMult */



  
  Matrix data;//, data_transpose;
  Matrix right_hand_side_e;

  data::Load("refined_astroset.ds", &data);
  data::Load("alldata_zs", &right_hand_side_e);
  /*
  if(data_transpose.n_cols() < data_transpose.n_rows()) {
    la::TransposeInit(data_transpose, &data);
  }
  else {
    data = data_transpose;
  }
  */
  

  
  // Communication stuff?
  Epetra_SerialComm comm;

  int row_length = data.n_cols(); // this should be correct despite obvious semantic conflicts

  // Required map for multivector stuff
  Epetra_BlockMap blockmap(1, row_length, 0, comm);

  printf("row_length = %d\n", row_length);
  



  // Define the linear problem.
  Epetra_MultiVector solution(blockmap, 1, true);
  Epetra_MultiVector right_hand_side(blockmap, 1, false);
  for(index_t j = 0; j < row_length; j++) {
    (*(right_hand_side(0)))[j] = right_hand_side_e.get(0, j);
  }


  SimpleLinearOperator simple_linear_operator(row_length, data, 1, 1,
					      &kernel_vector_mult);
  
  Epetra_LinearProblem linear_problem(&simple_linear_operator,
				      &solution, &right_hand_side);

  // Declare the iterative solver.
  AztecOO iterative_solver;

  simple_linear_operator.solver = &iterative_solver;

  iterative_solver.SetUserOperator(&simple_linear_operator);

  iterative_solver.SetProblem(linear_problem);

  int options[AZ_OPTIONS_SIZE];
  double params[AZ_PARAMS_SIZE];
  AZ_defaults(options, params);

  //iterative_solver.SetAllAztecOptions(options);
  //iterative_solver.SetAllAztecParams(params);
  
  iterative_solver.SetAztecOption(AZ_precond, AZ_none);
  //iterative_solver.SetAztecOption(AZ_precond, AZ_Jacobi);
  
  // Use Conjugate Gradient
  //iterative_solver.SetAztecOption(AZ_solver, AZ_cg);
  iterative_solver.SetAztecOption(AZ_solver, AZ_gmres);
  
  // Use modified Gram-Schmidt.
  //iterative_solver.SetAztecOption(AZ_orthog, AZ_modified);
  
  // No output.
  //iterative_solver.SetAztecOption(AZ_diagnostics, AZ_none);
  //iterative_solver.SetAztecOption(AZ_output, AZ_none);
  
  // Solve the linear system.
  iterative_solver.Iterate(row_length, 1.0E-9);
  

  //cout<<solution;



  // Finalize FastExec and print output results.
  fx_done();
  return 0;
}

