/**
 * @file cf.hpp
 * @author Mudit Raj Gupta
 *
 * Collaborative Filtering.
 *
 * Defines the CF class to perform Collaborative Filtering on the
 * specified data set.
 *
 */

#ifndef __MLPACK_METHODS_CF_CF_HPP
#define __MLPACK_METHODS_CF_CF_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/als/als.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <set>
#include <map>
#include <iostream>

namespace mlpack {
namespace cf /** Collaborative Filtering. */{

/**
 * This class implements Collaborative Filtering (CF). This 
 * implementation presently supports Alternating Least Squares 
 * for collaborative filtering. 
 * 
 * The template parameters can (optionally) be supplied are: the algorithm  
 * used for CF and the neighbourhood search for user similarity.
 *
 * A simple example of how to run Collaborative Filtering is shown below.
 *
 * @code
 * extern arma::mat data; // (user,item,rating) table
 * extern arma::Col<size_t> users; // users seeking recommendations
 * arma::mat recommendations; // Recommendations
 * size_t numRecommendations = 10;  
 *
 * CF<> cf(data); // Default options.
 *
 * //Default number of Recommendations for all users 
 * cf.GenerateRecommendations(recommendations);
 *
 * //Default number of Recommendations for specified users 
 * cf.GenerateRecommendations(recommendations, users);
 *
 * //10 Recommendations for specified users
 * cf.GenerateRecommendations(recommendations, users, numRecommendations);
 *
 * @endcode
 * 
 */

class CF
{ 
 public:
/**
  * Create a CF object and (optionally) set the parameters which CF
  * will be run with. 
  *
  * @param data Initial User,Item,Rating Matrix
  * @param numRecs Number of Recommendations for each user.
  * @param numUsersForSimilarity Size of the neighbourhood.
  */
  CF(const size_t numRecs,const size_t numUsersForSimilarity,
     arma::mat& data);
/**
  * Create a CF object and (optionally) set the parameters which CF
  * will be run with. 
  *
  * @param data Initial User,Item,Rating Matrix
  * @param numRecs Number of Recommendations for each user.
  */
  CF(const size_t numRecs, arma::mat& data);
/**
  * Create a CF object and (optionally) set the parameters which CF
  * will be run with. 
  *
  * @param data Initial User,Item,Rating Matrix
  */
  CF(arma::mat& data);
  //! Sets number of Recommendations.
  void NumRecs(size_t recs) 
  { 
    if (recs < 1)
    {
      Log::Warn << "CF::NumRecs(): invalid value (< 1) "
          "ignored." << std::endl;
      return;
    }
    this->numRecs = recs; 
  }

  //! Sets data
  void Data(arma::mat& d)
  {
    data = d;
  }
  
  //! Gets data
  arma::mat Data() 
  {
    return data;
  }

  //! Gets numRecs
  size_t NumRecs()
  { 
    return numRecs;
  }

  //! Sets number of user for calculating similarity.
  void NumUsersForSimilarity(size_t num) 
  { 
    if (num < 1)
    {
      Log::Warn << "CF::NumUsersForSimilarity(): invalid value (< 1) "
          "ignored." << std::endl;
      return;
    }
    this->numUsersForSimilarity = num; 
  }
  //! Gets number of users for calculating similarity/
  size_t NumUsersForSimilarity()
  {
    return numUsersForSimilarity;
  }
  
  //! Get the User Matrix.
  const arma::mat& W() const { return w; }
  //! Get the Item Matrix.
  const arma::mat& H() const { return h; }
  //! Get the Rating Matrix.
  const arma::mat& Rating() const { return rating; }

/*
 * Generates default number of recommendations for all users.
 *
 * @param recommendations Matrix to save recommendations
 */
  void GetRecommendations(arma::Mat<size_t>& recommendations);

/*
 * Generates default number of recommendations for specified users.
 *
 * @param recommendations Matrix to save recommendations
 * @param users Users for which recommendations are to be generated
 */   
  void GetRecommendations(arma::Mat<size_t>& recommendations, 
                          arma::Col<size_t>& users);

/*
 * Generates a fixed number of recommendations for specified users.
 *
 * @param recommendations Matrix to save recommendations
 * @param users Users for which recommendations are to be generated
 * @param num Number of Recommendations
 */
  void GetRecommendations(arma::Mat<size_t>& recommendations, 
                          arma::Col<size_t>& users, size_t num);

/*
 * Generates a fixed number of recommendations for specified users.
 *
 * @param recommendations Matrix to save recommendations
 * @param users Users for which recommendations are to be generated
 * @param num Number of Recommendations
 * @param neighbours Number of user to be considered while calculating     
 *        the neighbourhood
 */
  void GetRecommendations(arma::Mat<size_t>& recommendations, 
                          arma::Col<size_t>& users, size_t num, 
                          size_t neighbours);

 private:
  //! Number of Recommendations.
  size_t numRecs;
  //! Number of User for Similariy.
  size_t numUsersForSimilarity;
  //! User Matrix.
  arma::mat w;
  //! Item Matrix.
  arma::mat h;
  //! Rating Martix.
  arma::mat rating;
  //! Masking Matrix
  arma::mat mask;
  //! Initial Data Matrix.
  arma::mat data;
  //! Cleaned Data Matrix.
  arma::sp_mat cleanedData;
  //!Calculates a rating matrix with available data and approximations
  void CalculateApproximateRatings();
  //!Converts the User, Item, Value Matrix to User-Item Table
  void CleanData();
  //!Decomposes the cleanedData size_to user and item matrices
  void Decompose();
  //!Create ratings from user and item matrices
  void GenerateRating();
/*
 * Queries the obtained rating matrix.
 *
 * @param recommendations Matrix to save recommendations
 * @param users Users for which recommendations are to be generated
 */  
  void Query(arma::Mat<size_t>& recommendations,arma::Col<size_t>& users);
/*
 * Selects item preferences of the users 
 *
 * @param query Matrix to store the item preference of the user.
 * @param users Users for which recommendations are to be generated
 */
  void CreateQuery(arma::mat& query,arma::Col<size_t>& users) const;
/*
 * Generates the neighbourhood of users. 
 *
 * @param query Matrix to store the item preference of the user.
 * @param modifiedRating Matrix to store the Modified Matix.
 * @param neighbourhood Matrix to store user neighbourhood.
 */
  void GetNeighbourhood(arma::mat& query,
                        arma::Mat<size_t>& neighbourhood);
/*
 * Calculates the Average rating users would have given to 
 * unrated items based on their similarity with other users. 
 *
 * @param neighbourhood Matrix to store user neighbourhood.
 * @param averages stores the average rating for each item.
 */
  void CalculateAverage(arma::Mat<size_t>& neighbourhood, 
                        arma::mat& averages) const;
/*
 * Calculates the top recommendations given average rating 
 * for each user.
 *
 * @param neighbourhood Matrix to store user neighbourhood.
 * @param averages stores the average rating for each item.
 */ 
  void CalculateTopRecommendations(arma::Mat<size_t>& recommendations, 
                                   arma::mat& averages,
                                   arma::Col<size_t>& users) const;

}; // class CF

}; // namespace cf
}; // namespace mlpack

#endif
