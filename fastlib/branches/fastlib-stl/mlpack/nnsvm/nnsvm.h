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

#include <fastlib/fastlib.h>

#include "nnsmo.h"

#include <typeinfo>

PARAM(double, "c", "Undocumented", "nnsvm", 10.0, false);
PARAM(double, "eps", "Undocumented", "nnsvm", 1.0e-6, false);
PARAM_INT("max_iter", "Undocumented", "nnsvm", 1000);
PARAM(double, "sigma", "Undocumented", "nnsvm", 0.0, true);

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
  index_t GetTypeId()
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
  index_t GetTypeId()
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

/**
* Class for NNSVM
*/
template<typename TKernel>
class NNSVM
{
  public:
    typedef TKernel Kernel;

  private:
    struct NNSVM_MODELS
    {
      double thresh_; //negation of the intercept
      arma::vec sv_coef_; // the alpha vector
      arma::vec w_; // the weight vector
      index_t num_sv_; // number of support vectors
    };
    NNSVM_MODELS model_;

  struct NNSVM_PARAMETERS
  {
    TKernel kernel_;
    std::string kernelname_;
    index_t kerneltypeid_;
    double c_;
    index_t b_;
    double eps_; //tolerance
    index_t max_iter_; // maximum iterations
  };
  NNSVM_PARAMETERS param_; // same for every binary model

  arma::mat support_vectors_;
  index_t num_features_;

  public:
    void Init(const arma::mat& dataset, index_t n_classes);
    void InitTrain(const arma::mat& dataset, index_t n_classes);
    void SaveModel(std::string modelfilename);
    void LoadModel(arma::mat& testset, std::string modelfilename);
    index_t Classify(const arma::vec& vector);
    void BatchClassify(arma::mat& testset, std::string testlabelfilename);
    void LoadModelBatchClassify(arma::mat& testset, std::string modelfilename, std::string testlabelfilename);
};

#include "nnsvm_impl.h"

#endif
