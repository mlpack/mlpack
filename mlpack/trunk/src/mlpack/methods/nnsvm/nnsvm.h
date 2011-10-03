/**
 * @file nnsvm.h
 *
 * This head file contains functions for performing NNSVM training.
 * NNSMO algorithm is employed.
 *
 * @see nnsmo.h
 */
#ifndef U_NNSVM_NNSVM_H
#define U_NNSVM_NNSVM_H

#include <mlpack/core.h>
#include <typeinfo>

#include "nnsmo.h"

PARAM_DOUBLE("c", "Undocumented", "nnsvm", 10.0);
PARAM_DOUBLE("eps", "Undocumented", "nnsvm", 1.0e-6);
PARAM_INT("max_iter", "Undocumented", "nnsvm", 1000);
PARAM_DOUBLE("sigma", "Undocumented", "nnsvm", 0.0);

enum kernelEnumType
{
  ID_LINEAR,
  ID_GAUSSIAN,
  ID_END_OF_LIST
};

/**
* Class for Linear Kernel
*/
struct SVMLinearKernel
{
  void Init() {}

  void Copy(const SVMLinearKernel& other) {}

  /* Kernel value evaluation */
  double Eval(const arma::vec& a, const arma::vec& b) const
  {
    return dot(a, b);
  }
  /* Kernel name */
  void GetName(std::string& kname) {
    kname = "linear";
  }
  /* Get an type ID for kernel */
  size_t GetTypeId()
  {
    return ID_LINEAR;
  }
  /* Save kernel parameters to file */
  void SaveParam(FILE* fp)
  {
  }
};

/**
* Class for Gaussian RBF Kernel
*/
class SVMRBFKernel
{
  /* Init of kernel parameters */
  std::vector<double> kpara_; // kernel parameters
  void Init() { //TODO: NULL->node
    kpara_.reserve(2);
    kpara_[0] = mlpack::IO::GetParam<double>("nnsvm/sigma"); //sigma
    kpara_[1] = -1.0 / (2 * pow(kpara_[0], 2.0)); //gamma
  }
  /* Kernel name */
  void GetName(std::string& kname)
  {
    kname = "gaussian";
  }
  /* Get an type ID for kernel */
  size_t GetTypeId()
  {
    return ID_GAUSSIAN;
  }
  /* Kernel value evaluation */
  double Eval(const arma::vec& a, const arma::vec& b) const
  {
    arma::vec diff = b - a;
    double distance_squared = arma::dot(diff, diff);
    return exp(kpara_[1] * distance_squared);
  }
  /* Save kernel parameters to file */
  void SaveParam(FILE* fp)
  {
    fprintf(fp, "sigma %g\n", kpara_[0]);
    fprintf(fp, "gamma %g\n", kpara_[1]);
  }
};

struct nnsvm_model
{
  double thresh_; //negation of the intercept
  arma::vec sv_coef_; // the alpha vector
  arma::vec w_; // the weight vector
  size_t num_sv_; // number of support vectors
};

namespace mlpack {
namespace nnsvm {

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
    std::string kernelname_;
    size_t kerneltypeid_;
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
    void Init(const arma::mat& dataset, size_t n_classes, size_t c, size_t b, double eps, size_t max_iter);
    void InitTrain(const arma::mat& dataset, size_t n_classes);
    void InitTrain(const arma::mat& dataset, size_t n_classes, size_t c, size_t b, double eps, size_t max_iter);
    void SaveModel(std::string modelfilename);
    void LoadModel(arma::mat& testset, std::string modelfilename);
    size_t Classify(const arma::vec& vector);
    void BatchClassify(arma::mat& testset, std::string testlabelfilename);
    void LoadModelBatchClassify(arma::mat& testset, std::string modelfilename, std::string testlabelfilename);
    double getThreshold() { return model_.thresh_; }
    size_t getSupportVectorCount() { return model_.num_sv_; }
    const arma::vec getSupportVectorCoefficients() { return model_.sv_coef_; }
    const arma::vec getWeightVector() { return model_.w_; }
};

} // namespace nnsvm
} // namespace mlpack

#include "nnsvm_impl.h"

#endif
