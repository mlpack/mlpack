/** 
 * @file lcc.hpp
 * @author Nishant Mehta
 *
 * Definition of the LocalCoordinateCoding class, which performs the Local 
 * Coordinate Coding algorithm
 */

#ifndef __MLPACK_METHODS_LCC_LCC_HPP
#define __MLPACK_METHODS_LCC_LCC_HPP

//#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/methods/lars/lars.hpp>

namespace mlpack {
namespace lcc {

/**
 * An implementation of Local Coordinate Coding (LCC) that codes data which 
 * approximately lives on a manifold using a variation of l1-norm regularized 
 * sparse coding; in LCC, the penalty on the absolute value of each point's 
 * coefficient for each atom is weighted by the squared distance of that point 
 * to that atom. The paper is below. 
 * Let d be the number of dimensions in the original space, m the number of 
 * training points, and k the number of atoms in the dictionary (the dimension 
 * of the learned feature space). The training data X is a d-by-m matrix where 
 * each column is a point and each row is a dimension. The dictionary D is a 
 * d-by-k matrix, and the sparse codes matrix Z is a k-by-m matrix.
 * This program seeks to minimize the objective:
 * min_{D,Z} ||X - D Z||_{Fro}^2 
             + lambda sum_{i=1}^m sum_{j=1}^k dist(X_i,D_j)^2 Z_i^j
 * where lambda > 0.
 *
 * This problem is solved by an algorithm that alternates between a dictionary
 * learning step and a sparse coding step. The dictionary learning step updates
 * the dictionary D by solving a linear system (note that the objective is a
 * positive definite quadratic program). The sparse coding step involves 
 * solving a large number of weighted l1-norm regularized linear regression 
 * problems problems; this can be done efficiently using LARS, an algorithm 
 * that can solve the LASSO (paper below).
 *
 * The papers:
 *
 * @incollection{NIPS2009_0719,
 *   title = {Nonlinear Learning using Local Coordinate Coding},
 *   author = {Kai Yu and Tong Zhang and Yihong Gong},
 *   booktitle = {Advances in Neural Information Processing Systems 22},
 *   editor = {Y. Bengio and D. Schuurmans and J. Lafferty and C. K. I. Williams and A. Culotta},
 *   pages = {2223--2231},
 *   year = {2009}
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
 */
class LocalCoordinateCoding {

 public:
  //void Init(const arma::mat& matX, u32 nAtoms, double lambda);
  
  
  /**
   * Set the parameters to LocalCoordinateCoding.
   *
   * @param matX Data matrix
   * @param nAtoms Number of atoms in dictionary
   * @param lambda Regularization parameter for weighted l1-norm penalty
   */
  LocalCoordinateCoding(const arma::mat& matX, arma::u32 nAtoms, double lambda);
  
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
   * Run LCC
   *
   * @param nIterations Maximum number of iterations to run algorithm
   */
  void DoLCC(arma::u32 nIterations);

  /**
   * Sparse code each point via distance-weighted LARS
   */
  void OptimizeCode();
  
  /** 
   * Learn dictionary by solving linear systemx
   *
   * @param adjacencies Indices of entries (unrolled column by column) of 
   *    the coding matrix Z that are non-zero (the adjacency matrix for the 
   *    bipartite graph of points and atoms)
   */
  void OptimizeDictionary(arma::uvec adjacencies);

  /**
   * Compute objective function
   */  
  double Objective(arma::uvec adjacencies);


  // accessors, modifiers, printers
  
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
  double lambda;
  
};

void RemoveRows(const arma::mat& X, arma::uvec rows_to_remove, arma::mat& X_mod);


}; // namespace lcc
}; // namespace mlpack

#endif
