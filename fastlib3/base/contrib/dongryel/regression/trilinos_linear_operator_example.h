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
#include "multi_conjugate_gradient.h"

/** @brief The type of linear operator to be applied in iterative
 *         solver.
 */
template<typename TKernel, typename TPruneRule>
class KrylovLinearOperator: public virtual Epetra_Operator {

  // Declare friend class of this method.
  template<typename TKernel_, typename TPruneRule_>
  friend class KrylovLpr;

  FORBID_ACCIDENTAL_COPIES(KrylovLinearOperator);

 private:
  /** @brief The internal query tree type used for the computation. */
  typedef BinarySpaceTree< DHrectBound<2>, Matrix, KrylovLprQStat<TKernel> > 
    QueryTree;

  /** @brief The internal reference tree type used for the
   *         computation.
   */
  typedef BinarySpaceTree< DHrectBound<2>, Matrix, KrylovLprRStat<TKernel> > 
    ReferenceTree;

  /** @brief The query root.
   */
  QueryTree *qroot_;

  /** @brief The reference root.
   */
  ReferenceTree *rroot_;

  /** @brief The query dataset.
   */
  Matrix qset_;

  Vector rset_inv_norm_consts_;
  int row_length_;
  
  KrylovLpr<TKernel, TPruneRule> *krylov_lpr_;

 public:

  KrylovLinearOperator
  (ReferenceTree *rroot_in, const Matrix &qset,
   const Vector &rset_inv_norm_consts_in, 
   int row_length_in, KrylovLpr<TKernel, TPruneRule> *krylov_lpr_in) {
    
    rroot_ = rroot_in;
    rset_inv_norm_consts_.Alias(rset_inv_norm_consts_in);
    row_length_ = row_length_in;
    
    krylov_lpr_ = krylov_lpr_in;
    map_ = new Epetra_Map(row_length_, 0, comm_);
  }

  ~KrylovLinearOperator() {
    delete map_;
  }

  int SetUseTranspose(bool UseTranspose) {
    return -1;
  }

  int Apply(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const {
    
    Matrix vector_l, vector_e;
    Vector vector_used_error, vector_n_pruned;

    vector_l.Init(row_length_, X.NumVectors());
    vector_e.Init(row_length_, X.NumVectors());
    vector_used_error.Init(X.NumVectors());
    vector_n_pruned.Init(X.NumVectors());

    // Initialize the multivector to zero.
    Y.PutScalar(0);

    for(index_t d = 0; d < row_length_; d++) {
      krylov_lpr_->ComputeWeightedVectorSum_
	(qroot_, qset_, rset_inv_norm_consts_, d,
	 vector_l, vector_e, vector_used_error, vector_n_pruned);

      // Accumulate the product between the computed vector and each
      // scalar component of the X.
      for(index_t j = 0; j < row_length_; j++) {
	Y.Pointers()[0][j] += X.Pointers()[0][d] * vector_e.get(j, 0);
      }
    } // end of iterating over each component.

    return 0;
  }
  
  int ApplyInverse(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const {
    return -1;
  }
  
  double NormInf() const {
    return -1;
  }
  
  const char *Label() const {
    return "Dual-tree Krylov Operator";
  }

  bool UseTranspose() const {
    return false;
  }
  
  bool HasNormInf() const {
    return false;
  }
  
  const Epetra_Comm &Comm() const {
    return comm_;
  }

  const Epetra_Map &OperatorDomainMap() const {
    const Epetra_Map &map_reference = *map_;
    return map_reference;
  }

  const Epetra_Map &OperatorRangeMap() const {
    const Epetra_Map &map_reference = *map_;
    return map_reference;
  }
  
 public:

  AztecOO *solver_;
  Epetra_SerialComm comm_;
  Epetra_Map *map_;

};

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::SolveLinearProblems_
(const Matrix &qset, const Matrix &right_hand_sides_e,
 Matrix &solution_vectors_e) {
  
  // Communication stuff?
  Epetra_SerialComm comm;

  // Required map for multivector stuff
  Epetra_BlockMap blockmap(1, row_length_, 0, comm);
    
  // Solve the linear systems for each query point.
  for(index_t q = 0; q < qset.n_cols(); q++) {

    // Make each column query vector as the whole dataset.
    Vector q_col;
    qset.MakeColumnVector(q, &q_col);
    Vector q_col_copy;
    q_col_copy.Copy(q_col);
    Matrix qset_copy;
    qset_copy.AliasColVector(q_col_copy);
    
    // Construct the query tree.
    QueryTree *qroot = tree::MakeKdTreeMidpoint<QueryTree>
      (qset_copy, 20, NULL, NULL);

    KrylovLinearOperator<TKernel, TPruneRule> 
      krylov_linear_operator(rroot_, qset_copy, rset_inv_norm_consts_, 
			     row_length_, this);

    krylov_linear_operator.qroot_ = qroot;
    krylov_linear_operator.qset_.Alias(qset_copy);

    // Define the linear problem.
    Epetra_MultiVector solution(blockmap, 1, true);
    Epetra_MultiVector right_hand_side(blockmap, 1, false);
    for(index_t j = 0; j < row_length_; j++) {
      (*(right_hand_side(0)))[j] = right_hand_sides_e.get(j, q);
    }

    Epetra_LinearProblem linear_problem(&krylov_linear_operator, &solution, 
					&right_hand_side);

    // Declare the iterative solver.
    AztecOO iterative_solver;
    krylov_linear_operator.solver_ = &iterative_solver;
    iterative_solver.SetUserOperator(&krylov_linear_operator);
    iterative_solver.SetProblem(linear_problem);

    // No preconditioner - currently. To use-one, I need to
    // implemement an N-body based preconditioner.
    iterative_solver.SetAztecOption(AZ_precond, AZ_none);

    // Use Conjugate Gradient
    iterative_solver.SetAztecOption(AZ_solver, AZ_cg);

    // Use modified Gram-Schmidt.
    iterative_solver.SetAztecOption(AZ_orthog, AZ_modified);

    // No output.
    iterative_solver.SetAztecOption(AZ_diagnostics, AZ_none);
    iterative_solver.SetAztecOption(AZ_output, AZ_none);
    
    // Solve the linear system.
    iterative_solver.Iterate(row_length_, 1.0E-2);

    // Delete the query tree.
    delete qroot;

    // Copy the solution.
    for(index_t j = 0; j < row_length_; j++) {
      solution_vectors_e.set(j, q, (*(solution(0)))[j]);
    }
  }
}
