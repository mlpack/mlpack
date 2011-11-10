#include <mlpack/core.h>
namespace mlpack {
namespace pca {

class PCA {

	public:
		PCA();
		void Apply(arma::mat& coeff, arma::mat& score, arma::mat& data);

};
};
};
