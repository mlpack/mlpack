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


class SimpleLinearOperator: public virtual Epetra_Operator {
  FORBID_ACCIDENTAL_COPIES(SimpleLinearOperator);

private:
  

  int n_points_;
  int n_dims_;

  // used for a simple linear transformation operator data_^T * data  
  Matrix data_;
  //  Matrix K_;

  GaussianKernel gaussian_kernel_;
  double norm_constant_;

  double sigma_;


  
  
public:
  
  SimpleLinearOperator(Matrix data_in, double bandwidth_in, double sigma_in) {
    
    data_.Copy(data_in);

    //la::MulTransBInit(data_in, data_in, &K_);
    //data::Load("K.txt", &K_);
    
    n_dims_ = data_in.n_rows();
    n_points_ = data_in.n_cols();

    map = new Epetra_Map(n_points_, 0, comm);

    gaussian_kernel_.Init(bandwidth_in, n_dims_);
    norm_constant_ = gaussian_kernel_.CalcNormConstant(n_dims_);

    sigma_ = sigma_in;


  }

  ~SimpleLinearOperator() {
    delete map;
  }


  int SetUseTranspose(bool UseTranspose) {
    return false;
  }

  int Apply (const Epetra_MultiVector &X, Epetra_MultiVector &Y) const {

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
	
	K_i[j] = gaussian_kernel_.EvalUnnormOnSq(dist) / norm_constant_;
      }
      
      K_i[i] += sigma_; // add (sigma * I) term

      double sum = 0;
      for(int j = 0; j < n_points_; j++) {
	sum += K_i[j] * X[0][j];
      }
      
      Y.Pointers()[0][i] = sum;
    }

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

  
  Matrix data, data_transpose;
  Matrix right_hand_side_e;

  data::Load("data.txt", &data_transpose);
  data::Load("rhs.txt", &right_hand_side_e);

  if(data_transpose.n_cols() < data_transpose.n_rows()) {
    la::TransposeInit(data_transpose, &data);
  }
  else {
    data = data_transpose;
  }

  

  
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
    (*(right_hand_side(0)))[j] = right_hand_side_e.get(j, 0);
  }


  SimpleLinearOperator simple_linear_operator(data, 1, 1);
  
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
  iterative_solver.SetAztecOption(AZ_solver, AZ_cg);
  //iterative_solver.SetAztecOption(AZ_solver, AZ_gmres);
  
  // Use modified Gram-Schmidt.
  //iterative_solver.SetAztecOption(AZ_orthog, AZ_modified);
  
  // No output.
  //iterative_solver.SetAztecOption(AZ_diagnostics, AZ_none);
  //iterative_solver.SetAztecOption(AZ_output, AZ_none);
  
  // Solve the linear system.
  iterative_solver.Iterate(row_length, 1.0E-9);
  

  cout<<solution;



  // Finalize FastExec and print output results.
  fx_done();
  return 0;
}

