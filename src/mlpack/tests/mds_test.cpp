/**@file mds_test.cpp
 *
 *Test file for MDS class
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/manifold/mds.hpp>

//#define BOOST_TEST_DYN_LINK
//#define BOOST_TEST_MODULE MDStest

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"


BOOST_AUTO_TEST_SUITE(MDStest);

using namespace std;
using namespace mlpack;
using namespace mlpack::manifold;
using namespace arma;

//To match the MDS result with matlab's implementation. 

BOOST_AUTO_TEST_CASE(DimensionalityReductionTestMatlab){
	
	MDS m;
	arma::mat B;
	bool a_isnegative,b_isnegative;
	
	//Matlab's reduced dimensions. Have to be put manually.
	mat matlab("15.1721 17.4301;"
			   "-24.8743 2.7574;"
			   "9.7022 -20.1875");
	
	//Raw input data, this is not the dissimilarity matrix
	mat data("20 30 40;"
			 "1 3 13;"
			 "5 44 8");

	BOOST_REQUIRE_EQUAL(matlab.n_rows,data.n_rows);
	
	//Calculating the euclidean dissimilarity(default)
	B=m.dissimilarity(data);
	//Reducing the dimensions performing MDS
	B=m.reduce(B,2);
	//Flippping the eigenvectors in descending order of eigenvalues to match matlab's similar output.
	B=fliplr(B);
	
	BOOST_REQUIRE_EQUAL(matlab.n_cols,B.n_cols);

	//If vectors are in opposite directions as compared to matlab, reverse them.
	for(size_t i=0;i<B.n_cols;i++){
		a_isnegative=matlab(0,i)<0;
		b_isnegative=B(0,i)<0;
		if(a_isnegative!=b_isnegative)
			B.col(i)*=-1;	
	}


	for (size_t row = 0; row < B.n_rows; row++)
		for (size_t col = 0; col < B.n_cols; col++)
			BOOST_CHECK_CLOSE(B(row, col), matlab(row, col), 0.001);

}

/* Information about the Kruskal's stress can be found online. This will compare the dissimilarity matrix of the raw 
 * input data and the dissimilarity matrix formed after reducing the dimensions. They should approximately be equal 
 * to solve the purpose of MDS. This data is interpreted in the form of stress. 
 */
BOOST_AUTO_TEST_CASE(DetermineKruskalStress){
	
	MDS m;
	double num, den,k_stress;
	arma::mat B,dis1,dis2;
	mat matlab("15.1721 17.4301;"
			   "-24.8743 2.7574;"
			   "9.7022 -20.1875");
	
	mat data("20 30 40;"
			 "1 3 13;"
			 "5 44 8");
	//Dissimilarity of the input data
	B=m.dissimilarity(data);
	dis1=B;
	
	//Dissimilarity after reduced dimensions
	B=m.reduce(B,2);
	dis2=m.dissimilarity(B);
	
	//Calculating Kruskal's stress.
	num=accu(square(dis2-dis1));
	den=accu(square(dis1));
	k_stress=std::sqrt(num/den);

	/*This shows the goodness of fit defined by kruskal.
	 *Stress  Goodness of fit
	 *0.200       poor
	 *0.100       fair
	 *0.050       good
	 *0.025       excellent
	 *0.000       perfect
	 */
	
	BOOST_REQUIRE_MESSAGE(k_stress<0.025,"The goodness of fit is : good to fair"<<k_stress);
	BOOST_TEST_MESSAGE("The goodness of fit is : excellent, kruskal's stress: "<< k_stress);



}

BOOST_AUTO_TEST_SUITE_END();
