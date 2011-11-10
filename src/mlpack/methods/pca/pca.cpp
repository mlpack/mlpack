#include "pca.hpp"
#include <mlpack/core.h>

namespace mlpack {
namespace pca {

	PCA::PCA(){}

	void PCA::Apply(arma::mat& coeff, arma::mat& score, arma::mat& data)
	{
		arma::princomp(coeff, score, arma::trans(data));
		/*arma::vec eigval;
		arma::mat eigvec;
		arma::mat cov_mat = arma::cov(m_data);
		arma::eig_sym(eigval, eigvec, cov_mat);
		data_transformed = arma::trans(eigvec) * m_data;*/
	}
};
};
