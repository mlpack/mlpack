
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_linear_regression_linear_regression.hpp:

Program Listing for File linear_regression.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_linear_regression_linear_regression.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/linear_regression/linear_regression.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
   #define MLPACK_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace regression  {
   
   class LinearRegression
   {
    public:
     LinearRegression(const arma::mat& predictors,
                      const arma::rowvec& responses,
                      const double lambda = 0,
                      const bool intercept = true);
   
     LinearRegression(const arma::mat& predictors,
                      const arma::rowvec& responses,
                      const arma::rowvec& weights,
                      const double lambda = 0,
                      const bool intercept = true);
   
     LinearRegression() : lambda(0.0), intercept(true) { }
   
     double Train(const arma::mat& predictors,
                  const arma::rowvec& responses,
                  const bool intercept = true);
   
     double Train(const arma::mat& predictors,
                  const arma::rowvec& responses,
                  const arma::rowvec& weights,
                  const bool intercept = true);
   
     void Predict(const arma::mat& points, arma::rowvec& predictions) const;
   
     double ComputeError(const arma::mat& points,
                         const arma::rowvec& responses) const;
   
     const arma::vec& Parameters() const { return parameters; }
     arma::vec& Parameters() { return parameters; }
   
     double Lambda() const { return lambda; }
     double& Lambda() { return lambda; }
   
     bool Intercept() const { return intercept; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(parameters));
       ar(CEREAL_NVP(lambda));
       ar(CEREAL_NVP(intercept));
     }
   
    private:
     arma::vec parameters;
   
     double lambda;
   
     bool intercept;
   };
   
   } // namespace regression
   } // namespace mlpack
   
   #endif // MLPACK_METHODS_LINEAR_REGRESSION_HPP
