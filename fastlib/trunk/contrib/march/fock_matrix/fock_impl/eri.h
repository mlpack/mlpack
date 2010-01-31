#ifndef ERI_H
#define ERI_H

#include "fastlib/fastlib.h"
#include "../../libint/include/libint/libint.h"
#include "../../libint/include/libint/hrr_header.h"
#include "../../libint/include/libint/vrr_header.h"
#include "integral_tensor.h"

#define MAX_FAC 200

class ShellPair;
class BasisShell;

namespace eri {

  
  ///////////////////////// initialization ////////////////////////
  
  /**
   * Overall initializer for ERI's.  Computes the double factorials and 
   * initializes Libint with init_libint_base()
   *
   * Call this function before using any other functions in namespace eri
   *
   * Only needs to be called once per program.
   */
  void ERIInit();
  
  /**
   * Frees the double factorial array.  
   */
  void ERIFree();
  
  
  //////////////////////// Helpers //////////////////////////
  
  /**
   * Computes the number of functions in a shell of the given momentum.
   */
  index_t NumFunctions(int momentum);
  
  /**
   * Normalization function for higher momenta
   */
  double ComputeNormalization(double exp, int x_mom, int y_mom, int z_mom);
  
  
  double ComputeGPTCenter(const Vector& A_vec, double alpha_A, 
                          const Vector& B_vec, double alpha_B, Vector* p_vec);
  
  double ComputeShellOverlap(const BasisShell& shellA, const BasisShell& shellB);
  
  /**
   * @brief Used for prescreening-like prune in multi tree code
   */
  double ComputeShellOverlap(double AB_dist_sq, double exp_A, double exp_B);
  
  double GammaLn(double xx);
  void GammaSeries(double* gamser, double a, double x, double* gln);
  void GammaCF(double* gammcf, double a, double x, double* gln);
  double GammaP(double a, double x);
  
  void Compute_F(double* F, int n, double t);
  
  
  
  /**
   * The coefficient in the binomial expansion in the GPT.
   */
  double GPTCoefficient(int k, int l1, int l2, double PA_x, double PB_x);
  
  
  double ComputeOverlapIntegral(const Vector& center_A, double exp_A, int mom_A, 
                                const Vector& center_B, double exp_B, int mom_B);
  
  double OverlapCartesianFactor(int l1, int l2, double PA_x, double PB_x, 
                                double gamma);
  
  void ComputeOverlapIntegrals(const BasisShell& shellA, 
                               const BasisShell& shellB, 
                               Vector* integrals);
  
  double KineticCartesianFactors(int l1, int l2, int m1, int m2, int n1, int n2,
                                 const Vector& PA, const Vector& PB, double expA, 
                                 double expB);
  
  void ComputeKineticIntegrals(const BasisShell& shellA, 
                               const BasisShell& shellB,
                               Vector* integrals);
  
  double NuclearFactor(int l1, int l2, int m1, int m2, int n1, int n2, double gamma, 
                       const Vector& PA, const Vector& PB, const Vector& CP, 
                       const Vector& F);
  
  void ComputeNuclearIntegrals(const BasisShell& shellA, 
                               const BasisShell& shellB,
                               const Vector& Cvec, int nuclear_charge,
                               Vector* integrals);
  
  
  /**
   * Computes the maximum density entry among the four basis shells
   * for use in prescreening 
   */
  double DensityBound(ShellPair& shellA, ShellPair& shellB, 
                      const Matrix& density);
  
  /**
   * Computes the maximum density among the two shells for use in LinK
   */
  double DensityBound(const BasisShell& A_shell, const BasisShell& B_shell, 
                      const Matrix& density);
  
  /*
   double ComputeKineticIntegral(const Vector& center_A, double exp_A, int mom_A, 
   const Vector& center_B, double exp_B, int mom_B);
   
   double ComputeKineticIntegral(BasisShell& shellA, BasisShell& shellB);
   
   double ComputeNuclearIntegral(const Vector& center_A, double exp_A, int mom_A, 
   const Vector& center_B, double exp_B, int mom_B, 
   const Vector& nuclear_center, 
   int nuclear_charge);
   
   double ComputeNuclearIntegral(BasisShell& shellA, BasisShell& shellB, 
   const Vector& nuclear_center, int nuclear_mass);
   */
  
  /////////////////////////// constants //////////////////////////// 
  
  const double pow_pi_2point5 = pow(math::PI, 2.5);
  
  
    /**
   * Used to swap entries of the array list permutation.
   *
   * This should really go in the ArrayList class.
   */
  void ArrayListSwap(index_t ind1, index_t ind2, ArrayList<index_t>* perm);
  
  void ArrayListSwapPointers(index_t ind1, index_t ind2, 
                             ArrayList<BasisShell*>* list);
  
  /**
   * Returns the index of the given integral in the array returned from LIBINT.
   * indices holds the a, b, c, d indices and momenta holds the momenta
   */
  index_t IntegralIndex(ArrayList<index_t> indices, ArrayList<index_t> momenta);
  
  index_t IntegralIndex(index_t a_ind, int A_mom, index_t b_ind, int B_mom,
                        index_t c_ind, int C_mom, index_t d_ind, int D_mom);
  
  
  /**
   * For the given Libint index and total momentum, 
   * returns the x y and z momenta
   */
  //void BasisMomenta(index_t ind, index_t momentum, int* x_mom, int* y_mom, 
  //                  int* z_mom);
  
  /**
   * For the given Cartesian momenta, returns this functions index in 
   * the Libint order
   */
  //index_t BasisIndex(int x_mom, int y_mom, int z_mom, index_t total_mom);
  
  
    // not sure what this is for
  //double ComputeGPTCenter(Vector& A_vec, double alpha_A, Vector& B_vec, 
  //                        double alpha_B, Vector* p_vec);


  
  /**
   * Used to add the contracted integrals into the global matrix
   */
  void AddSubmatrix(const ArrayList<index_t>& rows,
                    const ArrayList<index_t>& cols,
                    const Matrix& submat, Matrix* out_mat);

  void AddSubmatrix(index_t row_begin, index_t row_count,
                    index_t col_begin, index_t col_count,
                    const Matrix& submat, Matrix* out_mat);
  

  ////////////////////////// External Integral Routines //////////////////
  
  /**
   * Computes the Schwartz factor Q_{i j} = (i j|i j)^1/2
   */
  double SchwartzBound(BasisShell& i_shell, BasisShell& j_shell);
  
  void ComputeShellIntegrals(BasisShell& mu_fun, 
                             BasisShell& nu_fun, 
                             BasisShell& rho_fun, 
                             BasisShell& sigma_fun,
                             IntegralTensor* integrals);
                               
  void ComputeShellIntegrals(ShellPair& AB_shell, 
                             ShellPair& CD_shell,
                             IntegralTensor* integrals);
                          
  
  ////////////////////////// Internal Integral Routines ///////////////
  
  /**
   * Call this function from outside.  
   *
   * It returns the permutation applied to the shells for use in reading the 
   * integrals.
   *
   * After calling this, reference the integrals using the permutations
   */
  void ComputeERI(const ArrayList<BasisShell*>& shells, double overlapAB, 
                  double overlapCD, IntegralTensor* integrals);
  
  /**
   * This currently assumes that the momenta obey the conditions:
   * A_mom >= B_mom
   * C_mom >= D_mom
   * A_mom + B_mom <= C_mom + D_mom
   *
   * IMPORTANT: must have called ERIInit() before calling this function
   */
  void ComputeERIInternal(const Vector& A_vec, double A_exp, int A_mom, double normA,
                          const Vector& B_vec, double B_exp, int B_mom, double normB,
                          const Vector& C_vec, double C_exp, int C_mom, double normC,
                          const Vector& D_vec, double D_exp, int D_mom, double normD,
                          double overlapAB, double overlapCD,
                          IntegralTensor* integrals);
  
  double* Libint_Eri(const Vector& A_vec, double A_exp, int A_mom, 
                     const Vector& B_vec, double B_exp, int B_mom,
                     const Vector& C_vec, double C_exp, int C_mom,
                     const Vector& D_vec, double D_exp, int D_mom,
                     double aux_fac, Libint_t* libint);
  
  
  ////////////////// Create Shells and ShellPairs ///////////////////////
  
  /**
   * Forms the list of BasisShells from the centers, exponents, and momenta
   *
   * Returns the total number of basis functions
   */
  index_t CreateShells(const Matrix& centers, const Vector& exponents, 
                    const Vector& momenta, ArrayList<BasisShell>* shells_out);
                            
  /**
   * Compute the list of significant shell pairs.
   *
   * Currently, shell pairs are screened by the size of their Schwartz factor, 
   * but some implementations may use overlap screening.   
   */
  index_t ComputeShellPairs(ArrayList<ShellPair>* shell_pairs, 
                            ArrayList<BasisShell>& shells_in, 
                            double shell_pair_cutoff, const Matrix& density);

  /**
   * Compute the list of significant shell pairs.  shell_max[i] is the 
   * largest Schwartz prescreening estimate for shell i.  This is used in the 
   * LinK algorithm. 
   *
   * Currently, shell pairs are screened by the size of their Schwartz factor, 
   * but some implementations may use overlap screening.   
   */
  index_t ComputeShellPairs(ArrayList<ShellPair>* shell_pairs, 
                            ArrayList<BasisShell>& shells_in, 
                            double shell_pair_cutoff, Vector* shell_max, 
                            ShellPair**** sigma_for_nu, 
                            ArrayList<index_t>* num_per_shell, 
                            const Matrix& density);
  

}


#include "basis_shell.h"
#include "shell_pair.h"


#endif
