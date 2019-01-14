/**
 * @file range_search_test.cpp
 * @author Niteya Shah
 *
 * Test mlpackMain() of range_search_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "RangeSearchMain";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/range_search/range_search_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct RangeSearchTestFixture
{
 public:

  RangeSearchTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }
  ~RangeSearchTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(RangeSearchMainTest, RangeSearchTestFixture);
/*
* Check that the correct output is returned for a small synthetic input
* case
*/
BOOST_AUTO_TEST_CASE(SyntheticRangeSearch)
{
  cout<<"Synthetic Test 1"<<endl;
  //Matrix Input is expected in this format
  arma::mat x={{0,3,3,4,3,1},{4,4,4,5,5,2},{0,1,2,2,3,3}};
  std::string distance_file="distances.csv";
  std::string neighbors_file="neighbors.csv";
  double min_v=0,max_v=3;
  vector<std::vector<size_t>> neighbor_val={{},{2,3,4},{1,3,4,5},{1,2,4},{1,2,3},
                                         {2}};
  vector<std::vector<double>> distance_val={{},{1,1.73205,2.23607},
                                           {1,1.41421,1.41421,3},
                                           {1.73205,1.41421,1.41421},
                                           {2.23607,1.41421,1.41421},
                                           {3}};
  SetInputParam("reference",std::move(x));
  //To prevent warning for lack of definition
  SetInputParam("min", min_v);
  SetInputParam("max", max_v);
  SetInputParam("distances_file",distance_file);
  SetInputParam("neighbors_file",neighbors_file);
  cout<<" 1.Parameters Set"<<endl;

  mlpackMain();

  cout<<" 2.Model Created"<<endl;
  math::Range r(min_v, max_v);
  vector<vector<size_t>> neighbors;
  vector<vector<double>> distances;
  CLI::GetParam<RSModel*>("output_model")->Search(r, neighbors, distances);
  cout<<" 3.Search Executed"<<endl;

  CheckMatrices(neighbors,neighbor_val,1e-5);
  CheckMatrices(distances,distance_val);
  cout<<" 4.Results Verified"<<endl;

  cout<<"Passed Synthetic Test 1"<<endl<<endl;
}

BOOST_AUTO_TEST_CASE(ParameterTesting)
{
  cout<<"Synthetic Test 2"<<endl;
  arma::mat query_data={{5,3,1},{4,2,4},{3,1,7}};
  arma::mat x={{0,3,3,4,3,1},{4,4,4,5,5,2},{0,1,2,2,3,3}};
  vector<std::vector<double>> distance_val={
                {2.82843,2.23607,1.73205,2.23607,4.47214},
                {3.74166,2,2.23607,3.31662,3.60555,2.82843},{4.58258,4.47214}};
  vector<std::vector<size_t>> neighbor_val={{1,2,3,4,5},{0,1,2,3,4,5},{4,5}};
  std::string distance_file="distances.csv";
  std::string neighbors_file="neighbors.csv";
  double min_v=0,max_v=5;

  SetInputParam("query",query_data);
  SetInputParam("reference",std::move(x));
  SetInputParam("min", min_v);
  SetInputParam("max", max_v);
  SetInputParam("distances_file",distance_file);
  SetInputParam("neighbors_file",neighbors_file);
  cout<<" 1.Parameters Set"<<endl;

  mlpackMain();

  cout<<" 2.Model Created"<<endl;
  math::Range r(min_v, max_v);
  vector<vector<size_t>> neighbors;
  vector<vector<double>> distances;
  CLI::GetParam<RSModel*>("output_model")->Search(std::move(query_data), r,
                                                  neighbors, distances);
  cout<<" 3.Search with Query Executed"<<endl;

  CheckMatrices(neighbors,neighbor_val,1e-5);
  CheckMatrices(distances,distance_val);

  cout<<" 4.Results Verified"<<endl;

  cout<<"Passed Synthetic Test 2"<<endl<<endl;
}

BOOST_AUTO_TEST_CASE(ModelCheck)
{//check if an output model can be used again for a different usage
  cout<<"Model Reusability Test"<<endl;
  arma::mat input_data;
  double min_v=0,max_v=3;
  std::string dis="distances.csv";
  std::string neigh="neighbors.csv";
  data::Load("iris.csv",input_data);
  SetInputParam("reference",std::move(input_data));
  SetInputParam("min", min_v);
  SetInputParam("max", max_v);
  SetInputParam("distances_file",dis);
  SetInputParam("neighbors_file",neigh);
  cout<<" 1.Parameters Set"<<endl;

  mlpackMain();

  arma::mat query_data;
  data::Load("iris_test.csv",query_data);
  RSModel* output_model=std::move(CLI::GetParam<RSModel*>("output_model"));
  CLI::GetSingleton().Parameters()["reference"].wasPassed=false;
  cout<<" 2.Model Created and copied"<<endl;

  SetInputParam("input_model",std::move(output_model));
  SetInputParam("query",std::move(query_data));

  mlpackMain();

  cout<<" 3.Model Reinserted"<<endl;
  if(!(output_model==CLI::GetParam<RSModel*>("output_model")))
  {
    BOOST_FAIL("Models are not Equal");
  }
  else
  {
    cout<<"Model Checking Test Passed"<<endl<<endl;
  }
}

BOOST_AUTO_TEST_CASE(LeafValueTesting)
{
  cout<<"Leaf Value Tests"<<endl;
  //Testing 3 different leaf values - default 20, 15 and 25
  //Ensure that results match for different leaf values
  arma::mat x={{0,3,3,4,3,1},{4,4,4,5,5,2},{0,1,2,2,3,3}};
  std::string distance_file="distances.csv";
  std::string neighbors_file="neighbors.csv";
  double min_v=0,max_v=3;
  vector<std::vector<int>> neighbor_val={{},{2,3,4},{1,3,4,5},{1,2,4},{1,2,3},
                                         {2}};
  vector<std::vector<float>> distance_val={{},{1,1.73205,2.23607},
                                           {1,1.41421,1.41421,3},
                                           {1.73205,1.41421,1.41421},
                                           {2.23607,1.41421,1.41421},
                                           {3}};
  math::Range r(min_v, max_v);
  vector<vector<size_t>> neighbors,neighbors_temp;
  vector<vector<double>> distances,distances_temp;
  vector<int> arr{20,15,25};
  SetInputParam("reference",x);
  //To prevent warning for lack of definition
  SetInputParam("min", min_v);
  SetInputParam("max", max_v);
  SetInputParam("distances_file",distance_file);
  SetInputParam("neighbors_file",neighbors_file);
  SetInputParam("leaf_size",arr[0]);
  cout<<" Setting Base size for testing :"<<arr[0]<<endl;
  //Default leaf size is 20

  mlpackMain();

  RSModel* output_model1=std::move(CLI::GetParam<RSModel*>("output_model"));

  output_model1->Search(r,neighbors,distances);

  bindings::tests::CleanMemory();

  for(size_t i=1;i<arr.size();i++)
  {
    cout<<"  Testing for Leaf Size :"<<arr[i]<<endl;
    SetInputParam("leaf_size",arr[i]);
    SetInputParam("reference",x);
      //To prevent warning for lack of definition
    SetInputParam("min", min_v);
    SetInputParam("max", max_v);
    SetInputParam("distances_file",distance_file);
    SetInputParam("neighbors_file",neighbors_file);

    mlpackMain();

    RSModel* output_model2=std::move(CLI::GetParam<RSModel*>("output_model"));
    output_model2->Search(r,neighbors_temp,distances_temp);

    CheckMatrices(neighbors,neighbors_temp,1e-5);
    CheckMatrices(distances,distances_temp);
  }
  cout<<"Leaf value Test Passed"<<endl<<endl;
}

//All trees should give the same results for fixed input parameters
BOOST_AUTO_TEST_CASE(TreeTypeTesting)
{
  cout<<"Tree Type Testing"<<endl;
  std::string distance_file="distances.csv";
  std::string neighbors_file="neighbors.csv";
  double min_v=0,max_v=3;
  arma::mat query_data,input_data;
  vector<vector<size_t>> neighbors,neighbors_temp;
  vector<vector<double>> distances,distances_temp;
  std::vector<string> trees={"kd","cover","r","r-star","ball","x","hilbert-r","r-plus"
                              ,"r-plus-plus","vp","rp","max-rp","ub","oct"};

  data::Load("iris.csv",input_data);
  data::Load("iris_test.csv",query_data);
  math::Range r(min_v, max_v);
  //Define Base with kd Tree
  SetInputParam("tree_type",trees[0]);
  SetInputParam("min", min_v);
  SetInputParam("max", max_v);
  SetInputParam("distances_file",distance_file);
  SetInputParam("neighbors_file",neighbors_file);
  SetInputParam("reference",std::move(input_data));
  SetInputParam("query",query_data);

  mlpackMain();

  cout<<"  Created Base value with kd tree"<<endl;
  CLI::GetParam<RSModel*>("output_model")->Search(std::move(query_data),r,neighbors,distances);

  for(size_t i=1;i<trees.size();i++)
  {
    bindings::tests::CleanMemory();
    cout<<"  Testing for Tree type :"<<trees[i]<<endl;
    data::Load("iris.csv",input_data);
    data::Load("iris_test.csv",query_data);
    math::Range r(min_v, max_v);

    SetInputParam("min", min_v);
    SetInputParam("max", max_v);
    SetInputParam("distances_file",distance_file);
    SetInputParam("neighbors_file",neighbors_file);
    SetInputParam("reference",std::move(input_data));
    SetInputParam("query",query_data);
    SetInputParam("tree_type",trees[i]);

    mlpackMain();

    CLI::GetParam<RSModel*>("output_model")->Search(std::move(query_data),r,neighbors_temp,distances_temp);
    CheckMatrices(neighbors,neighbors_temp);
    CheckMatrices(distances,distances_temp);
    cout<<"    Successful"<<endl;

  }
}

BOOST_AUTO_TEST_SUITE_END();
