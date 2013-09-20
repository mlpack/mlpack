/**
 * @file cf_test.cpp
 * @author Mudit Raj Gupta
 *
 * Test file for CF class.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/cf/cf.hpp>
#include <iostream>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(CFTest);

using namespace mlpack;
using namespace mlpack::cf;
using namespace std;

/**
 * Make sure that the rated item is not recommended
 */
BOOST_AUTO_TEST_CASE(CFConstructorTest)
{
  //Matrix to save data
  arma::mat dataset, retDataset; 

  //Loading GroupLens data 
  data::Load("GroupLens100k.csv", dataset);

  //Number of Recommendations
  size_t numRecs = 10;
  
  //Number of users for similarity
  size_t numUsersForSimilarity = 5;

  //Creating a CF object
  CF c(numRecs, numUsersForSimilarity, dataset);
  
  //Getter
  retDataset = c.Data();
  //Checking for parameters
  BOOST_REQUIRE_EQUAL(c.NumRecs(), numRecs);
  BOOST_REQUIRE_EQUAL(c.NumUsersForSimilarity(), numUsersForSimilarity);
  
  //Checking Data
  BOOST_REQUIRE_EQUAL(retDataset.n_rows, dataset.n_rows);  
  BOOST_REQUIRE_EQUAL(retDataset.n_cols, dataset.n_cols);    

  //Checking Values
  for (size_t i=0;i<dataset.n_rows;i++)
    for (size_t j=0;j<dataset.n_cols;j++)
      BOOST_REQUIRE_EQUAL(retDataset(i,j),dataset(i,j));  
}

/*
 * Make sure that correct number of recommendations 
 * are generated when query set. Default Case.
 */
BOOST_AUTO_TEST_CASE(CFGetRecommendationsAllUsersTest)
{
  //Dummy number of recommendations
  size_t numRecs = 3;
  //GroupLens100k.csv dataset has 943 users
  size_t numUsers = 943;
 
  //Matrix to save recommednations
  arma::Mat<size_t> recommendations;
  
  //Matrix to save data
  arma::mat dataset; 

  //Loading GroupLens data 
  data::Load("GroupLens100k.csv", dataset);

  //Creating a CF object
  CF c(dataset);

  //Setting Number of Recommendations
  c.NumRecs(numRecs);
  
  //Generating Recommendations when query set is not specified 
  c.GetRecommendations(recommendations);

  //Checking if correct number of Recommendations are generated
  BOOST_REQUIRE_EQUAL(recommendations.n_rows, numRecs);

  //Checking if recommendations are generated for all users 
  BOOST_REQUIRE_EQUAL(recommendations.n_cols, numUsers);
}

/*
 * Make sure that the recommendations are genrated 
 * for queried users only
 */ 
BOOST_AUTO_TEST_CASE(CFGetRecommendationsQueriedUserTest)
{
  //Number of users for which recommendations are seeked
  size_t numUsers = 10;
  
  //Default number of recommendations
  size_t numRecsDefault = 5;
 
  //Creaating dummy query set
  arma::Col<size_t> users = arma::zeros<arma::Col<size_t> >(numUsers,1);
  for (size_t i=0;i<numUsers;i++)
    users(i) = i+1;
  
  //Matrix to save recommednations
  arma::Mat<size_t> recommendations;
  
  //Matrix to save data
  arma::mat dataset; 

  //Loading GroupLens data 
  data::Load("GroupLens100k.csv", dataset);

  //Creating a CF object
  CF c(dataset);
  
  //Generating Recommendations when query set is not specified 
  c.GetRecommendations(recommendations, users);

  //Checking if correct number of Recommendations are generated
  BOOST_REQUIRE_EQUAL(recommendations.n_rows, numRecsDefault);

  //Checking if recommendations are generated for all users 
  BOOST_REQUIRE_EQUAL(recommendations.n_cols, numUsers);
}

BOOST_AUTO_TEST_SUITE_END();
