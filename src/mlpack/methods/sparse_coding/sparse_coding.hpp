/** 
 * @file sparse_coding.hpp
 * @author Nishant Mehta
 *
 * Definition of the SparseCoding class, which performs l1 (LASSO) or 
 * l1+l2 (Elastic Net)-regularized sparse coding with dictionary learning
 */

#ifndef __MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_HPP
#define __MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_HPP

//#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/methods/lars/lars.hpp>

namespace mlpack {
namespace sparse_coding {

/**
 * An implementation of Sparse Coding with Dictionary Learning that achieves 
 * sparsity via an l1-norm regularizer on the codes (LASSO) or an (l1+l2)-norm 
 * regularizer on the codes (the Elastic Net).
 * Let d be the number of dimensions in the original space, m the number of 
 * training points, and k the number of atoms in the dictionary (the dimension 
 * of the learned feature space). The training data X is a d-by-m matrix where 
 * each column is a point and each row is a dimension. The dictionary D is a 
 * d-by-k matrix, and the sparse codes matrix Z is a k-by-m matrix.
 * This program seeks to minimize the objective:
 * min_{D,Z} 0.5 ||X - D Z||_{Fro}^2\ + lambda_1 sum_{i=1}^m ||Z_i||_1
 *                                    + 0.5 lambda_2 sum_{i=1}^m ||Z_i||_2^2
 * subject to ||D_j||_2 <= 1 for 1 <= j <= k
 * where typically lambda_1 > 0 and lambda_2 = 0.
 *
 * This problem is solved by an algorithm that alternates between a dictionary
 * learning step and a sparse coding step. The dictionary learning step updates 
 * the dictionary D using a Newton method based on the Lagrange dual (see the 
 * paper below for details). The sparse coding step involves solving a large 
 * number of sparse linear regression problems; this can be done efficiently 
 * using LARS, an algorithm that can solve the LASSO or the Elastic Net (papers below).
 *
 * Here are those papers:
 *
 * @code
 * @incollection{lee2007efficient,
 *   title = {Efficient sparse coding algorithms},
 *   author = {Honglak Lee and Alexis Battle and Rajat Raina and Andrew Y. Ng},
 *   booktitle = {Advances in Neural Information Processing Systems 19},
 *   editor = {B. Sch\"{o}lkopf and J. Platt and T. Hoffman},
 *   publisher = {MIT Press},
 *   address = {Cambridge, MA},
 *   pages = {801--808},
 *   year = {2007}
 * }
 * @endcode
 *
 * @code
 * @article{efron2004least,
 *   title={Least angle regression},
 *   author={Efron, B. and Hastie, T. and Johnstone, I. and Tibshirani, R.},
 *   journal={The Annals of statistics},
 *   volume={32},
 *   number={2},
 *   pages={407--499},
 *   year={2004},
 *   publisher={Institute of Mathematical Statistics}
 * }
 * @endcode
 *
 * @code
 * @article{zou2005regularization,
 *   title={Regularization and variable selection via the elastic net},
 *   author={Zou, H. and Hastie, T.},
 *   journal={Journal of the Royal Statistical Society Series B},
 *   volume={67},
 *   number={2},
 *   pages={301--320},
 *   year={2005},
 *   publisher={Royal Statistical Society}
 * }
 * @endcode
 */
class SparseCoding {

 public:
  // void Init(double* memX, u32 nDims, u32 nPoints,
  // 	    u32 nAtoms, double lambda1);

  //void SetDictionary(double* memD);

  
  /**
   * Set the parameters to SparseCoding. lambda2 defaults to 0.
   *
   * @param matX Data matrix
   * @param nAtoms Number of atoms in dictionary
   * @param lambda1 Regularization parameter for l1-norm penalty
   * @param lambda2 Regularization parameter for l2-norm penalty
   */
  SparseCoding(const arma::mat& matX, arma::u32 nAtoms, double lambda1, double lambda2 = 0);
  

  /**
   * Initialize dictionary somehow
   */
  void InitDictionary();
  
  /** 
   * Load dictionary from a file
   * 
   * @param dictionaryFilename Filename containing dictionary
   */
  void LoadDictionary(const char* dictionaryFilename);
  
  /**
   * Initialize dictionary by drawing k vectors uniformly at random from the 
   * unit sphere
   */
  void RandomInitDictionary();

  /**
   * Initialize dictionary by initializing each atom to a normalized mixture of
   * a small number of randomly selected points in X
   */
  void DataDependentRandomInitDictionary();

  /**
   * Initialize an atom to a normalized mixture of a small number of randomly
   * selected points in X
   *
   * @param atom The atom to initialize
   */
  void RandomAtom(arma::vec& atom);

  
  // core algorithm functions

  /**
   * Run Sparse Coding with Dictionary Learning
   *
   * @param nIterations Maximum number of iterations to run algorithm
   */
  void DoSparseCoding(arma::u32 nIterations);

  /**
   * Sparse code each point via LARS
   */
  void OptimizeCode();
  
  /** 
   * Learn dictionary via Newton method based on Lagrange dual
   *
   * @param adjacencies Indices of entries (unrolled column by column) of 
   *    the coding matrix Z that are non-zero (the adjacency matrix for the 
   *    bipartite graph of points and atoms)
   */
  void OptimizeDictionary(arma::uvec adjacencies);

  /**
   * Project each atom of the dictionary onto the unit ball
   */
  void ProjectDictionary();

  /**
   * Compute objective function
   */
  double Objective();


  // accessors, modifiers, printers

  //! Modifier for matX
  void SetData(const arma::mat& matX);

  //! Modifier for matD
  void SetDictionary(const arma::mat& matD);
  
  //! Accessor for matD
  const arma::mat& MatD() {
    return matD;
  }
  
  //! Accessor for matZ
  const arma::mat& MatZ() {
    return matZ;
  }

  // Print the dictionary matD
  void PrintDictionary();

  // Print the sparse codes matZ
  void PrintCoding();

 private:
  arma::u32 nDims;
  arma::u32 nAtoms;
  arma::u32 nPoints;

  // data (columns are points)
  arma::mat matX;

  // dictionary (columns are atoms)
  arma::mat matD; 
  
  // sparse codes (columns are points)
  arma::mat matZ; 
  
  // l1 regularization term
  double lambda1; 
  
  // l2 regularization term
  double lambda2; 
  
};

void RemoveRows(const arma::mat& X, arma::uvec rows_to_remove, arma::mat& X_mod);


}; // namespace sparse_coding
}; // namespace mlpack

#endif
