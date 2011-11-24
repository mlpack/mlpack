/**
 * @file nnsvm.hpp
 *
 * This head file contains functions for performing NNSVM training.
 * NNSMO algorithm is employed.
 *
 * @see nnsmo.hpp
 */
#ifndef __MLPACK_METHODS_NNSVM_NNSVM_HPP
#define __MLPACK_METHODS_NNSVM_NNSVM_HPP

#include <mlpack/core.h>
#include <typeinfo>

#include "nnsmo.hpp"

PARAM_DOUBLE("c", "Undocumented", "nnsvm", 10.0);
PARAM_DOUBLE("eps", "Undocumented", "nnsvm", 1.0e-6);
PARAM_INT("max_iter", "Undocumented", "nnsvm", 1000);
PARAM_DOUBLE("sigma", "Undocumented", "nnsvm", 0.0);

namespace mlpack {
namespace nnsvm {

struct nnsvm_model
{
  double thresh_; //negation of the intercept
  arma::vec sv_coef_; // the alpha vector
  arma::vec w_; // the weight vector
  size_t num_sv_; // number of support vectors
};

/**
* Class for NNSVM
*/
template<typename TKernel>
class NNSVM
{
 public:
  typedef TKernel Kernel;

 private:
  struct nnsvm_model model_;

  struct NNSVM_PARAMETERS
  {
    TKernel kernel_;
    double c_;
    size_t b_;
    double eps_; //tolerance
    size_t max_iter_; // maximum iterations
  };
  NNSVM_PARAMETERS param_; // same for every binary model

  arma::mat support_vectors_;
  size_t num_features_;

 public:
  void Init(const arma::mat& dataset, size_t n_classes);
  void Init(const arma::mat& dataset,
            size_t n_classes,
            size_t c,
            size_t b,
            double eps,
            size_t max_iter);
  void InitTrain(const arma::mat& dataset, size_t n_classes);
  void InitTrain(const arma::mat& dataset,
                 size_t n_classes,
                 size_t c,
                 size_t b,
                 double eps,
                 size_t max_iter);
  void SaveModel(std::string modelfilename);
  void LoadModel(arma::mat& testset, std::string modelfilename);
  size_t Classify(const arma::vec& vector);
  void BatchClassify(arma::mat& testset, std::string testlabelfilename);
  void LoadModelBatchClassify(arma::mat& testset,
                              std::string modelfilename,
                              std::string testlabelfilename);
  double getThreshold() { return model_.thresh_; }
  size_t getSupportVectorCount() { return model_.num_sv_; }
  const arma::vec getSupportVectorCoefficients() { return model_.sv_coef_; }
  const arma::vec getWeightVector() { return model_.w_; }
};

}; // namespace nnsvm
}; // namespace mlpack

#include "nnsvm_impl.hpp"

#endif // __MLPACK_METHODS_NNSVM_NNSVM_HPP
