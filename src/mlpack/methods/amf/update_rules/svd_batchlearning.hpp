#ifndef __MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCHLEARNING_HPP
#define __MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCHLEARNING_HPP

#include <mlpack/core.hpp>

namespace mlpack
{
namespace amf
{
class SVDBatchLearning
{
public:
    SVDBatchLearning(double u = 0.000001,
                     double kw = 0,
                     double kh = 0,
                     double min = -DBL_MIN,
                     double max = DBL_MAX)
        : u(u), kw(kw), kh(kh), min(min), max(max) {}

    /**
    * The update rule for the basis matrix W.
    * The function takes in all the matrices and only changes the
    * value of the W matrix.
    *
    * @param V Input matrix to be factorized.
    * @param W Basis matrix to be updated.
    * @param H Encoding matrix.
    */
    template<typename MatType>
    inline void WUpdate(const MatType& V,
                               arma::mat& W,
                               const arma::mat& H) const
    {
        size_t n = V.n_rows;
        size_t m = V.n_cols;

        size_t r = W.n_cols;

        arma::mat deltaW(n, r);
        deltaW.zeros();

        for(size_t i = 0; i < n; i++)
        {
            for(size_t j = 0; j < m; j++)
                if(V(i,j) != 0) deltaW.row(i) += (V(i,j) - Predict(W.row(i), H.col(j))) * arma::trans(H.col(j));
            deltaW.row(i) -= kw * W.row(i);
        }

        W += u * deltaW;
    }

    /**
    * The update rule for the encoding matrix H.
    * The function takes in all the matrices and only changes the
    * value of the H matrix.
    *
    * @param V Input matrix to be factorized.
    * @param W Basis matrix.
    * @param H Encoding matrix to be updated.
    */
    template<typename MatType>
    inline void HUpdate(const MatType& V,
                               const arma::mat& W,
                               arma::mat& H) const
    {
        size_t n = V.n_rows;
        size_t m = V.n_cols;

        size_t r = W.n_cols;

        arma::mat deltaH(r, m);
        deltaH.zeros();

        for(size_t j = 0; j < m; j++)
        {
            for(size_t i = 0; i < n; i++)
                if(V(i,j) != 0) deltaH.col(j) += (V(i,j) - Predict(W.row(i), H.col(j))) * arma::trans(W.row(i));
            deltaH.col(j) -= kh * H.col(j);
        }

        H += u*deltaH;
    }
private:

    double Predict(const arma::mat& wi, const arma::mat& hj) const
    {
        arma::mat temp = (wi * hj);
        double out = temp(0,0);
        return out;
    }

    double u;
    double kw;
    double kh;
    double min;
    double max;
};
} // namespace amf
} // namespace mlpack


#endif

