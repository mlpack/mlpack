/**
 * @file mds.cpp
 * @author Dhawal Arora
 * Defines the functions for the MDS class to perform Multi-Dimensional Scaling on the
 * specified data set.
 */

#include "mds.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::manifold;
using namespace arma;

//Currently all the computations are done considering "metric" MDS, although MDS type variable has been included
//to change to non-metric later. Default is metric.

//create MDS object and compute MDS by calling dissimilarity() and reduce()
MDS::MDS(const arma::mat& data,
		 size_t dimensionality,
		 std::string dis/*= "euclidean"*/,
		 std::string type/*="metric"*/){
	
	arma::mat d;
	if(!dis.compare("euclidean")){
		d=dissimilarity(data);
		reduce(d, dimensionality);	
	}
}


/*this will take dissimilarity matrix and compute MDS by reducing the data into specified reduced dimensions.
 * The implementation here is done based on this paper:
 * http://www.math.uwaterloo.ca/~aghodsib/courses/f06stat890/readings/tutorial_stat890.pdf
 * and the test examples can be taken from here:
 * https://homepages.uni-tuebingen.de/florian.wickelmaier/pubs/Wickelmaier2003SQRU.pdf
 */
const arma::mat& MDS::reduce(const arma::mat& dissimalarity_mat,
							 size_t dimensionality,
							 std::string dis/*= "euclidean"*/,
							 std::string type/*="metric"*/){
	

	arma::mat XX,H,I,one,eigvec;
	arma::vec eigval;
	double n=dissimalarity_mat.n_rows;
	I=eye<mat>(n,n);
	one=ones<mat>(n,n);
	H=I-1/n*one;
	XX=-0.5*H*square(dissimalarity_mat)*H;
	eig_sym(eigval,eigvec,XX);
	
	if(dimensionality<eigvec.n_rows){
		eigval.shed_rows(0,eigval.n_elem-dimensionality-1);
		eigvec.shed_cols(0,eigvec.n_cols-dimensionality-1);
	}
	
	//If dissimilarity matrix is somethinig else than euclidean,
	// an extra step for making negative eigenvalues to 0 might be neeeded as mentioned in pg6 of this paper.
	//http://www.lcayton.com/resexam.pdf

	if(!dis.compare("euclidean")){
		XX=eigvec*sqrt(diagmat(eigval));
		transformed_data=XX;
	}
	return transformed_data;

}


//This calls the suitable distance function e.g euclidean_dissim here in case of euclidean distance.
//Currently support for euclidean dissimilarity is given which can be extended later for other dissimilarity functions.

arma::mat MDS::dissimilarity(const arma::mat& data, std::string dis/*="euclidean"*/){
	arma::mat d;
	if(!dis.compare("euclidean"))
		d=euclidean_dissim(data);
	return d;
}

/*This calculates the euclidean dissimilarity between X and Y vectors using 
 *X.X+Y.Y-2*X.Y
 *This along with other armadillo functions gives the fastest implementation for calculating nxn euclidean 
 *dissimilarity matrix with only one matrix product required in the whole implementation 
 */

arma::mat MDS::euclidean_dissim(arma::mat data){
	arma::mat d;
	arma::vec v=sum(square(data),1);
	d=repmat(v,1,data.n_rows);
	d=d+trans(d)-2*data*trans(data);
	return sqrt(d);
}

