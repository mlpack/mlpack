/**
 * @file cf.cpp
 * @author Mudit Raj Gupta
 *
 * Collaborative Filtering.
 *
 * Implementation of CF class to perform Collaborative Filtering on the
 * specified data set.
 *
 */
#include "cf.hpp"
#include <mlpack/methods/nmf/nmf.hpp>
#include <mlpack/methods/nmf/als_update_rules.hpp>

using namespace mlpack::nmf;
using namespace std;

namespace mlpack {
namespace cf {

/**
 * Construct the CF object.
 */
CF::CF(arma::mat& data) :
     data(data)
{
  Log::Info<<"Constructor (param: input data, default: numRecs;neighbourhood)"<<endl;
  this->numRecs = 5;
  this->numUsersForSimilarity = 5;
}

CF::CF(const size_t numRecs,arma::mat& data) :
     data(data)
{
  // Validate number of recommendation factor.
  if (numRecs < 1)
  {
    Log::Warn << "CF::CF(): number of recommendations shoud be > 0("
        << numRecs << " given). Setting value to 5.\n";
    //Setting Default Value of 5
    this->numRecs = 5;
  }
  else
    this->numRecs = numRecs;
  this->numUsersForSimilarity = 5;
}

CF::CF(const size_t numRecs, const size_t numUsersForSimilarity,
     arma::mat& data) :
     data(data)
{
  // Validate number of recommendation factor.
  if (numRecs < 1)
  {
    Log::Warn << "CF::CF(): number of recommendations shoud be > 0("
        << numRecs << " given). Setting value to 5.\n";
    //Setting Default Value of 5
    this->numRecs = 5;
  }
  else
    this->numRecs = numRecs;
  // Validate neighbourhood size.
  if (numUsersForSimilarity < 1)
  {
    Log::Warn << "CF::CF(): neighbourhood size shoud be > 0("
        << numUsersForSimilarity << " given). Setting value to 5.\n";
    //Setting Default Value of 5
    this->numUsersForSimilarity = 5;
  }
  else
    this->numUsersForSimilarity = numUsersForSimilarity;
}

void CF::CalculateApproximateRatings()
{
  Log::Info<<"CalculatineApproximateRating"<<endl;
  //Build the initial rating tables with missing values
  //if(cleanedData.n_rows==0)
  CleanData();
  //Decompose the size_tiial table size_to user and item
  Decompose();
  //Generate new table by multiplying approximate values
  GenerateRating();
}

void CF::GetRecommendations(arma::Mat<size_t>& recommendations)
{
  // Build the initial rating tables with missing values.
  CleanData();
  // Used to save user IDs.
  arma::Col<size_t> users =
    arma::zeros<arma::Col<size_t> >(cleanedData.n_cols, 1);
  // Getting all user IDs.
  for (size_t i = 0; i < cleanedData.n_cols; i++)
    users(i) = i + 1;

  // Calling base function for recommendations.
  GetRecommendations(recommendations, users);
}

void CF::GetRecommendations(arma::Mat<size_t>& recommendations,
                            arma::Col<size_t>& users)
{
  // Base function for calculating recommendations.
  // Operations independent of the query.
  CalculateApproximateRatings();
  // Query-dependent operations.
  Query(recommendations,users);
}

void CF::GetRecommendations(arma::Mat<size_t>& recommendations,
                            arma::Col<size_t>& users,size_t num)
{
  //Setting Number of Recommendations
  NumRecs(num);
  //Calling Base Function for Recommendations
  GetRecommendations(recommendations,users);
}

void CF::GetRecommendations(arma::Mat<size_t>& recommendations,
                            arma::Col<size_t>& users,size_t num,size_t s)
{
  //Setting number of users that should be used for calculating
  //neighbours
  NumUsersForSimilarity(s);
  //Setting Number of Recommendations
  NumRecs(num);
  //Calling Base Function for Recommendations
  GetRecommendations(recommendations,users,num);
}

void CF::CleanData()
{
  Log::Info<<"CleanData";
  //Temporarily stores max user id
  double maxUserID;
  //Temporarily stores max item id
  double maxItemID;
  //Calculating max users and items
  maxUserID = data(0,0);
  maxItemID = data(1,0);
  for (size_t i=1;i<data.n_cols;i++)
  {
      if(data(0,i)>maxUserID)
        maxUserID = data(0,i);
      if(data(1,i)>maxItemID)
        maxItemID = data(1,i);
  }
  //Temporarily stores sparcely populated rating matrix
  arma::sp_mat tmp((size_t)maxItemID,(size_t)maxUserID);
  //Temporarily stores mask matrix
  arma::mat lMask = arma::ones<arma::mat>((size_t)maxItemID,
                                          (size_t)maxUserID);
  //Calculates the initial User-Item table
  for (size_t i=0;i<data.n_cols;i++)
    tmp(data(1,i)-1,data(0,i)-1) = data(2,i);
  //Mask
  for (size_t i=0;i<data.n_cols;i++)
    lMask(data(1,i)-1,data(0,i)-1) = -1.0;
  //Storing in a global variable
  cleanedData = tmp;
  //Storing the mask
  mask=lMask;
 //data::Save("cleanedData.csv",cleanedData);
 //data::Save("mask.csv",mask);
}

void CF::Decompose()
{
   Log::Info<<"Decompose"<<endl;
   size_t rank = 2;
  //Presenltly only ALS is supported as an Optimizer
  //Should be converted to a template
  NMF<RandomInitialization, WAlternatingLeastSquaresRule,
      HAlternatingLeastSquaresRule> als(10000, 1e-5);
  als.Apply(cleanedData,rank,w,h);
  //data::Save("w.csv",w);
  //data::Save("h.csv",h);
}

void CF::GenerateRating()
{
  Log::Info<<"GenerateRatings"<<endl;
  //Calculating approximate rating
  rating = w*h;
  //data::Save("rating.csv",rating);
}

void CF::Query(arma::Mat<size_t>& recommendations,
               arma::Col<size_t>& users)
{
  Log::Info<<"Query"<<endl;
  //Temproraily stores feature vector of queried users
  arma::mat query(rating.n_rows, users.n_rows);// = arma::zeros<arma::mat>(rating.n_rows, users.n_rows);
  //Calculates the feature vector of queried users
  CreateQuery(query,users);
  //Temporary storage for neighbourhood of the queried users
  arma::Mat<size_t> neighbourhood;
  //Calculates the neighbourhood of the queried users
  GetNeighbourhood(query,neighbourhood);
  //Temporary storage for storing the average rating for each
  //user in their neighbourhood
  arma::mat averages = arma::zeros<arma::mat>(rating.n_rows,query.n_cols);
  //Calculates the average values
  CalculateAverage(neighbourhood,averages);
  //Calculates the top recommendations
  CalculateTopRecommendations(recommendations,averages,users);
}

void CF::CreateQuery(arma::mat& query,arma::Col<size_t>& users) const
{
  Log::Info<<"CreateQuery"<<endl;
  //Selecting feature vectors of queried users
  for(size_t i=0;i<users.n_rows;i++)
    for(size_t j=0;j<rating.col(i).n_rows;j++)
      query(j,i) = rating(j,users(i)-1);
  //data::Save("query.csv",query);
}

void CF::GetNeighbourhood(arma::mat& query,
                         arma::Mat<size_t>& neighbourhood)
{
  Log::Info<<"GetNeighbourhood"<<endl;
  if(numUsersForSimilarity>rating.n_cols)
  {
    Log::Warn << "CF::GetNeighbourhood(arma::mat,armaMat<size_t>):"
        <<"neighbourhood size should be > total number of users("
        << rating.n_cols << " given). Setting value to number of users.\n";
    NumUsersForSimilarity(rating.n_cols);
  }
  //Creating an Alknn object
  //Should be moved to templates
  AllkNN a(rating, query);
  //Temproraily storing distance between neighbours
  arma::mat resultingDistances;
  //Building neighbourhood
  a.Search(numUsersForSimilarity, neighbourhood,
           resultingDistances);
  //data::Save("neighbourhood.csv",neighbourhood);
}

void CF::CalculateAverage(arma::Mat<size_t>& neighbourhood,
                      arma::mat& averages) const
{
  Log::Info<<"CalculateAverage"<<endl;
  //Temprorary Storage for calculating sum
  arma::Col<double> tmp = arma::zeros<arma::Col<double> >(rating.n_rows,1);
  size_t j;
  //Iterating over all users
  for(size_t i=0;i<neighbourhood.n_cols;i++)
  {
    tmp = arma::zeros<arma::Col<double> >(rating.n_rows,1);
    //Iterating over all neighbours
    for(j=0;j<neighbourhood.n_rows;j++)
      tmp += rating.col(neighbourhood(j,i));
    //Calculating averages
    averages.col(i) = tmp/j;
  }
  //data::Save("averages.csv",averages);
}

void CF::CalculateTopRecommendations(arma::Mat<size_t>& recommendations,
                                 arma::mat& averages,
                                 arma::Col<size_t>& users) const
{
  Log::Info<<"CalculateTopRecommendations"<<endl;
  int recos = numRecs;
  if(averages.n_cols<numRecs)
    recos = averages.n_rows;
  //Stores recommendations
  arma::Mat<size_t> rec = arma::zeros<arma::Mat<size_t> >(recos,users.n_rows);
  //Stores valid ratings for items by a user
  arma::Col<double> tmp = arma::zeros<arma::Col<double> >(rating.n_cols,1);
  //Maps the items to their ratings
  std::map<double,size_t> tmpMap;
  //Iterator for the Item-Rating Map
  std::map<double,size_t>::reverse_iterator iter;
  std::map<double,size_t>::iterator it;
  //Keeps count of number of recommendations provided for a user
  size_t count;
  //Iterate for all users
  for(size_t i=0;i<users.n_rows;i++)
  {
    count=0;
    //Dot product between average rating and mask to dilute the ratings
    // of the items that user i has already rated
    tmp = averages.col(users(i)-1) % mask.col(users(i)-1);
    //Mapping Rating to Items
    for(size_t j=0;j<tmp.n_rows;j++)
      if(tmp(j)>=0)
        tmpMap.insert(std::pair<double,size_t>(tmp(j),j+1));
    //Iterating over Item-Rating Map
    for(iter=tmpMap.rbegin();iter!=tmpMap.rend();++iter)
    {
       //Saving recommendations to the recommendations table
      rec(count,i) = (size_t)iter->second;
      count++;
      //break is desired number of recommendations are saved
      if(count==numRecs)
        break;
    }
    //Removing the items from the map
    //note: Item 0 is just to maintain the consistency and it
    //represents not recommendations were available
    for(it=tmpMap.begin();it!=tmpMap.end();++it)
      tmpMap.erase(it);
  }
  //Saving to recommendations
  recommendations = rec;
}

}; // namespace mlpack
}; // namespace cf
