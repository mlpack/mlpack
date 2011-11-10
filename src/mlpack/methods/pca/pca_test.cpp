#include "pca.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::pca;

int main(int argc, char *argv[])
{
	int n_rows;
	int n_cols;

	mat data;
	data	<< 0.8402 << 0.7984 << 0.3352 << endr
			<< 0.3944 << 0.9116 << 0.7682 << endr
			<< 0.7831 << 0.1976 << 0.2778 << endr;

	mat coeff_actual;

	coeff_actual	<< -0.41721 << -0.610904 <<  0.672854 << endr
					<<  0.743785 << -0.654959 << -0.133446 << endr
					<<  0.522226 <<  0.444775 <<  0.727636 << endr;

	mat score_actual;

	score_actual	<< -0.0144 << -0.2645 << 0 << endr
					<<  0.4819 <<  0.1262 << 0 << endr
					<< -0.4675 <<  0.1383 << 0 << endr;

	mat coeff;
	mat score;


	mat trans_data = arma::trans(data);
	mlpack::pca::PCA p;
	p.Apply(coeff, score, trans_data);

	n_rows = data.n_rows;
	n_cols = data.n_cols;

	/*cout << "test matrix : " << endl << data;

	cout << "coeff : " << endl << coeff;

	cout << "score : " << endl << score;*/

	for(int i = 0; i < n_rows; i++)
	{
		for(int j = 0; j < n_cols; j++)
		{
			assert(fabs(coeff_actual(i, j) - coeff(i, j)) < 0.0001);
			assert(fabs(score_actual(i, j) - score(i, j)) < 0.0001);
		}
	}

	return 0;
}
