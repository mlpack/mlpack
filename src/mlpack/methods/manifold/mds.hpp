/**
 * @file mds.hpp
 * @author Dhawal Arora
 * Defines the MDS class to perform Multi-Dimensional Scaling on the
 * specified data set.
 */
#ifndef __MLPACK_METHODS_MANIFOLD_MDS_HPP
#define __MLPACK_METHODS_MANIFOLD_MDS_HPP

#include <mlpack/core.hpp>
namespace mlpack{
namespace manifold{
/**
 *This class implements Multi-Dimensional Scaling.MDS is a classical approach that maps the original high dimensional
 *space to a lower dimensional space, but does so in an attempt to preserve pairwise dis-
 *tances between all the data points in both the dimensions. Further information on MDS can be found through research 
 *papers on the internet.Some other libraries also have robust implemenation of MDS. 
 */

class MDS{

public: 

/**
 *This creates an MDS object without specifying any parameters. Further computations for MDS then are done using
 *other functions provided in the class separtely.
 */
 	MDS(){}

/**
 * This creates an MDS object and simultaneously solves the MDS by calling dissimalirity() and reduce() function of this class.
 * class. The final reduced vectors are stored in transformed_data matrix.
 *
 *@param data It is the coordinate matrix nxm where n is the number of objects and m is
 *			  the initial dimensionality of the data.
 *@param dimensionality It is the number of dimensions in which the data has to be reduced.
 *@param dis It is a string which specifies the type of dissimilarity matrix to be computed for the dataset.
 *		     The default is the euclidean matrix where Xij is the euclidean distance between vector Xi and Xj.
 *@param type It specifies type of MDS. It is either mertic or non-metric.
 */

	MDS(const arma::mat& data,
		size_t dimensionality,
		std::string dis= "euclidean",
		std::string type="metric");

/**
 * This function calculates the dissimilarity matrix, provided the input dataset. Dissimilarity is the measure of
 * distance between two data points.The default dissimilarity distance is the euclidean distance. This returns the 
 * euclidean matrix where Xij is the euclidean distance between vector Xi and  Xj. 
 *
 * @param data It is the coordinate matrix nxm where n is the number of objects and m is
 *			   the initial dimensionality of the data.
 * @param dis  It is a string which specifies the type of dissimilarity matrix to be computed for the dataset.
 */
	arma::mat dissimilarity(const arma::mat& data, std::string dis="euclidean");

/**
 * This function performs the MDS algorithm and returns dxm matrix where d is the number of datapoints
 * and m is the reduced dimension.
 *
 *@param dissimilarity_mat It is the dissimilarity matrix.
 *@param dimensionality It is the number of dimensions in which the data has to be reduced.
 *@param dis It is a string which specifies the type of dissimilarity matrix to be computed for the dataset.
 *		     The default is the euclidean matrix where Xij is the euclidean distance between vector Xi and Xj.
 *@param type It specifies type of MDS. It is either mertic or non-metric.
 */

	const arma::mat& reduce(const arma::mat& dissimalarity_mat,
							size_t dimensionality,
							std::string dis= "euclidean", 
							std::string type="metric");

	//const arma::mat& reduce(std::string type="metric", size_t dimensionality);

	//It calculates the euclidean dissimilarity matrix.
	arma::mat euclidean_dissim(arma::mat data);


	//This returns the final reduced matrix of dimension dxm where d is the number of data points 
	//and m is the reduced dimension.
	const arma::mat& transformedData() const { return transformed_data; }

	//arma::mat& transformedData() { return transformed_data; }


private:
	arma::mat transformed_data;
	//arma::mat dissimalarity_matrix;
	//std::string dis_type;
};//class MDS

};//namespace manifold
};//namespace mlpack

#endif	
