#ifndef SOLVELINSYS_H
#define SOLVELINSYS_H


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
  //float** K_;
  //Matrix K_;

  GaussianKernel kernel_;
  //double norm_constant_;

  double sigma_squared_;

  KernelVectorMult* kernel_vector_mult_;


  
  
public:
  
  SimpleLinearOperator(int n_points_in, Matrix data_in, 
		       double bandwidth_in, double sigma_squared_in,
		       KernelVectorMult* kernel_vector_mult_in) {

    n_points_ = n_points_in;
    kernel_vector_mult_ = kernel_vector_mult_in;

    
    data_.Copy(data_in);



    




    
    n_dims_ = data_in.n_rows();
    //n_points_ = data_in.n_cols();



    map = new Epetra_Map(n_points_, 0, comm);

    //kernel_.Init(bandwidth_in, n_dims_);
    //double norm_constant = kernel_.CalcNormConstant(n_dims_);
    //printf("norm_constant = %f\n", norm_constant_);

    sigma_squared_ = sigma_squared_in;

    /*
    // for debugging purposes, explicitly represent kernel matrix K

    if((K_ = (float**) malloc(n_points_ * sizeof(float*))) == NULL) {
      printf("failed\n");
      exit(1);
    }

    for(int i = 0; i < n_points_; i++) {
      if((K_[i] = (float*) malloc(n_points_ * sizeof(float))) == NULL) {
	printf("failed on %d\n", i);
	exit(1);
      }
    }

    printf("start writing\n");
    for(int i = 0; i < n_points_; i++) {
      for(int j = 0; j < n_points_; j++) {
	K_[i][j] = 0;
      }
    }
    printf("done writing\n");


    for(int i = 0; i < n_points_; i++) {
      Vector v_i;
      data_.MakeColumnVector(i, &v_i);
      
      for(int j = 0; j < n_points_; j++) {
	Vector v_j;
	data_.MakeColumnVector(j, &v_j);
	
	double dist = la::DistanceSqEuclidean(v_i, v_j);
	K_[i][j] = kernel_.EvalUnnormOnSq(dist) / norm_constant;
      }
    }

    for(int i = 0; i < n_points_; i++) {
      K_[i][i] += sigma_squared_;
    }

    Matrix K_mat;
    K_mat.Init(n_points_, n_points_);
    for(int i = 0; i < n_points_; i++) {
      for(int j = 0; j < n_points_; j++) {
	K_mat.set(j, i, K_[i][j]);
      }
    }

    data::Save("K.txt", K_mat);
    


    printf("done computing kernel matrix\n\n\n\n");
    */

    //const char *K_file_name = "K.txt";
    //data::Save(K_file_name, K_);

    
//     int errors = 0;
//     for(int i = 0; i < n_points_; i++) {
//       for(int j = 0; j < n_points_; j++) {
// 	if(K_.get(i,j) != K_.get(j,i)) {
// 	  errors++;
// 	}
//       }
//     }
    
//     printf("sigma_squared_ = %f\n", sigma_squared_);
//     printf("errors = %d\n", errors);
    

    // end debugging explicit representation of K
    


  }

  ~SimpleLinearOperator() {
    delete map;
  }


  int SetUseTranspose(bool UseTranspose) {
    return false;
  }

  int Apply (const Epetra_MultiVector &X, Epetra_MultiVector &Y) const {
    
        
    // FAST SUMMATION CODE
    Vector weights_vector;
    Vector results;
    weights_vector.Init(n_points_);
    results.Init(n_points_);


    for(int j = 0; j < n_points_; j++) {
      weights_vector[j] = X.Pointers()[0][j];
    }

    kernel_vector_mult_ -> Reset();
    kernel_vector_mult_ -> ComputeKernelMatrixVectorMultiplication(weights_vector, &results);

    //printf("\n");
    for(int i = 0; i < n_points_; i++) {
      Y.Pointers()[0][i] = results[i] + (sigma_squared_ * weights_vector[i]);
      //results[i] += sigma_squared_ * weights_vector[i];
      //printf("%f ", Y.Pointers()[0][i]);//results[i]);
    }
    //printf("\n");
    
    // END FAST SUMMATION CODE
    

    /*
    printf("data.n_rows() = %d\ndata.n_cols() = %d\n",
	   data_.n_rows(),
	   data_.n_cols());
    */
    
    /*
    Vector K_i;
    K_i.Init(n_points_);


    for(int i = 0; i < n_points_; i++) {
      Vector v_i;
      data_.MakeColumnVector(i, &v_i);
      
      for(int j = 0; j < n_points_; j++) {
	Vector v_j;
	data_.MakeColumnVector(j, &v_j);
	
	//double dist = la::Dot(v_i, v_j);
	//K_i[j] = dist;

	double dist = la::DistanceSqEuclidean(v_i, v_j);
	K_i[j] = epan_kernel_.EvalUnnormOnSq(dist);
      }

      la::Scale(1 / norm_constant_, &K_i);

      
      K_i[i] += sigma_squared_; // add (sigma^2 * I) term

      double sum = 0;
      for(int j = 0; j < n_points_; j++) {
	sum += (K_i[j] * X.Pointers()[0][j]);
      }
      
      Y.Pointers()[0][i] = sum;
    }
    */    

    /*
    // explicit K
    Vector x_vec;
    x_vec.Init(n_points_);

    Vector y_vec;
    y_vec.Init(n_points_);

    for(int i = 0; i < n_points_; i++) {
      x_vec[i] = X.Pointers()[0][i];
      printf("%f\t%f\n", X.Pointers()[0][i], x_vec[i]);
    }
    //exit(1);

    for(int i = 0; i < n_points_; i++) {
      double sum = 0;
      for(int j = 0; j < n_points_; j++) {
	sum += K_[i][j] * (float)x_vec[j];
      }
      y_vec[i] = sum;
    }

    for(int i = 0; i < n_points_; i++) {
      Y.Pointers()[0][i] = y_vec[i];
    }
    
    // end explicit K
    
    //   x_vec.PrintDebug("y_vec");
    */

    /*
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



// A(r) x = rhs
// A(r) is some linear operator derived from reference points r
void SolveLinearSystem(Matrix references, Vector rhs, double bandwidth, double sigma_squared, Vector* solution) {

  DEBUG_ONLY(printf("references.n_rows() = %d\nreferences.n_cols() = %d\n",
		    references.n_rows(), references.n_cols()));
  
  
  KernelVectorMult kernel_vector_mult;
  
  struct datanode* kernel_vector_mult_module =
    fx_submodule(NULL, "kernel_vector_mult", "kernel_vector_mult_module");


  DEBUG_ONLY(printf("sigma_squared = %f\n", sigma_squared));

  kernel_vector_mult.Init(references, bandwidth, kernel_vector_mult_module);
  
    
  

  
  // Communication stuff?
  Epetra_SerialComm comm;

  int row_length = references.n_cols(); // this should be correct despite obvious semantic conflicts

  // Required map for multivector stuff
  Epetra_BlockMap blockmap(1, row_length, 0, comm);

  DEBUG_ONLY(printf("row_length = %d\n", row_length));
  



  // Define the linear problem.
  Epetra_MultiVector solution_e(blockmap, 1, true);
  Epetra_MultiVector right_hand_side_e(blockmap, 1, false);
  for(index_t j = 0; j < row_length; j++) {
    (*(right_hand_side_e(0)))[j] = rhs[j];
  }


  SimpleLinearOperator simple_linear_operator(row_length, references,
					      bandwidth, sigma_squared,
					      &kernel_vector_mult);
  
  Epetra_LinearProblem linear_problem(&simple_linear_operator,
				      &solution_e, &right_hand_side_e);

  // Declare the iterative solver.
  AztecOO iterative_solver;

  simple_linear_operator.solver = &iterative_solver;

  iterative_solver.SetUserOperator(&simple_linear_operator);

  iterative_solver.SetProblem(linear_problem);

  int options[AZ_OPTIONS_SIZE];
  double params[AZ_PARAMS_SIZE];
  AZ_defaults(options, params);

  iterative_solver.SetAllAztecOptions(options);
  iterative_solver.SetAllAztecParams(params);
  
  iterative_solver.SetAztecOption(AZ_precond, AZ_none);
  //iterative_solver.SetAztecOption(AZ_precond, AZ_Jacobi);
  
  // Use Conjugate Gradient
  iterative_solver.SetAztecOption(AZ_solver, AZ_cg);
  //iterative_solver.SetAztecOption(AZ_solver, AZ_gmres);
  //iterative_solver.SetAztecOption(AZ_solver, AZ_cg_condnum);
  
  // Use modified Gram-Schmidt.
  iterative_solver.SetAztecOption(AZ_orthog, AZ_modified);
  
  // No output.
  //iterative_solver.SetAztecOption(AZ_diagnostics, AZ_none);
  //iterative_solver.SetAztecOption(AZ_output, AZ_none);
  
  // Solve the linear system.
  iterative_solver.Iterate(row_length, 1.0E-6);
  
  
  solution -> Init(row_length);
  for(index_t j = 0; j < row_length; j++) {
    (*solution)[j] = solution_e.Pointers()[0][j];
  }

}

#endif /* SOLVELINSYS_H */
