#ifndef SOFTMAX_FINETUNE_HPP
#define SOFTMAX_FINETUNE_HPP

namespace mlpack {
namespace nn {

class SoftmaxFineTune
{
 public:
  template<typename T>
  static void LastGradient(arma::mat const &input,
                           arma::mat const &parameters,
                           T const &model,
                           arma::mat &gradient)
  {
    gradient = -(parameters.t() * (model.GroundTruth() - model.Probabilities())) /
               static_cast<double>(input.n_cols);
    gradient = gradient % (input % (1 - input));
  }

  static void Gradient(arma::mat const &input,
                       arma::mat const &parameters,
                       arma::mat const &deriv,
                       arma::mat &output)
  {
    output = (parameters.t() * deriv) % (input % (1 - input));
  }

};

} // namespace nn
} // namespace mlpack

#endif // SOFTMAX_FINETUNE_HPP
