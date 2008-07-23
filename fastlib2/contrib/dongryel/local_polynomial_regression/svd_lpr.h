/** @author Dongryeol Lee (dongryel)
 *
 *  This header file declares function prototypes for computing local
 *  polynomial regression using a batch variant of singular value
 *  decomposition.
 *
 *  @bug None.
 */

#ifndef SVD_LPR_H
#define SVD_LPR_H

/** @brief A computation class for local polynomial regression.
 *
 */

template<typename TKernel>
class SvdLpr {

  // Do not copy this class object using a naive copy constructor!
  FORBID_ACCIDENTAL_COPIES(SvdLpr);

 private:
  
  /** @brief The reference dataset used to build the local polynomial
   *         regression model.
   */
  Matrix reference_set_;
  
 public:

  
};

#endif
