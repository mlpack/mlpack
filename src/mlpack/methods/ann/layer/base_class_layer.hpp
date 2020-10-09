#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann {

template<typename eT>
class BaseLayer
{
 public:

  virtual void Forward(const arma::Mat<eT>&,
                       arma::Mat<eT>&) = 0;

  virtual void Backward(const arma::Mat<eT>&,
                const arma::Mat<eT>&,
                arma::Mat<eT>&) = 0;

  virtual void Gradient(const arma::Mat<eT>&,
                const arma::Mat<eT>&,
                arma::Mat<eT>&) = 0;
};

}
}