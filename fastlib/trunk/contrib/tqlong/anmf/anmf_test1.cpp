#include <fastlib/fastlib.h>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "anmf.h"
#include "allnn_kdtree_distance_matrix.h"
#include "allnn_auction_max_weight_matching.h"

//namespace po = boost::program_options;
using namespace std;
using namespace boost::posix_time;
//using namespace boost::program_options;

boost::program_options::options_description desc("Allowed options");
boost::program_options::variables_map vm;

void process_options(int argc, char** argv)
{
  desc.add_options()
      ("help", "produce help message")
      ("reference", boost::program_options::value<string>()->default_value("reference.txt") ,"file consists of reference points")
      ("query", boost::program_options::value<string>()->default_value("query.txt") ,"file consists of query points")
      ("matrix", boost::program_options::value<string>()->default_value("matrix.txt") ,"file consists of weight matrix")
      ("random", boost::program_options::value<int>()->default_value(0) ,"generate random reference and query points")
      ("dim", boost::program_options::value<int>()->default_value(2) ,"generate random reference and query points")
      ("maxRange", boost::program_options::value<double>()->default_value(100) ,"max range of random points");

  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (vm.count("help"))
  {
    cout << desc << endl;
    exit(1);
  }
}

void generateRandom(const char* refFile, const char* queryFile)
{
  int n_points = vm["random"].as<int>();
  int dim = vm["dim"].as<int>();
  Matrix ref, query;
  ref.Init(dim, n_points);
  query.Init(dim, n_points);

  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < n_points; j++)
    {
      ref.ref(i, j) = math::Random(0, vm["maxRange"].as<double>());
      query.ref(i, j) = math::Random(0, vm["maxRange"].as<double>());
    }
  }
  data::Save(refFile, ref);
  data::Save(queryFile, query);
}

int main(int argc, char** argv)
{
  process_options(argc, argv);

  if (vm["random"].as<int>() > 0)
  {
    generateRandom(vm["reference"].as<string>().c_str(), vm["query"].as<string>().c_str());
  }

  ptime time_start(second_clock::local_time());

//  if (vm.count("query"))
//  {
//    Matrix query;
//    data::Load(vm["query"].as<string>().c_str(), &query);

//    anmf::KDNode qRoot(query, std::vector<double>(query.n_cols(), 0.0));

//    int i;
//    double min = std::numeric_limits<double>::infinity();
//    Vector x;
//    x.Init(2);
//    x[0] = 1.3; x[1] = 1.3;
//    qRoot.randomBound(x, i, min);
//    qRoot.nearestNeighbor(x, i, min);

////    cout << qRoot.toString() << "\n";

//    cout << "min = " << min << " i = " << i << "\n";
//  }

  if (vm.count("reference") && vm.count("query"))
  {
    typedef anmf::KDTreeDistanceMatrix DistanceMatrix;
//    typedef anmf::NaiveDistanceMatrix DistanceMatrix;

    Matrix ref, query;
    data::Load(vm["reference"].as<string>().c_str(), &ref);
    data::Load(vm["query"].as<string>().c_str(), &query);

    DistanceMatrix M(ref, query);
    anmf::AuctionMaxWeightMatching<DistanceMatrix> matcher(M, true);
    for (int i = 0; i < M.n_rows(); i++)
    {
      Vector r_i, q_j;
      int j = matcher.leftMatch(i);
      M.row(i, r_i);
      M.col(j, q_j);
      cout << "ref " << anmf::toString(r_i) << " query " << anmf::toString(q_j) << "\n";
    }
//    for (int refIndex = 0; refIndex < ref.n_cols(); refIndex++)
//      cout << "reference point " << refIndex << " --> query point " << matcher.leftMatch(refIndex) << endl;
  }

//  if (vm.count("matrix"))
//  {
//    Matrix M;
//    data::Load(vm["matrix"].as<string>().c_str(), &M);

//    anmf::AuctionMaxWeightMatching<Matrix> matcher(M, true, true);
//    for (int bidder = 0; bidder < M.n_rows(); bidder++)
//      cout << "bidder " << bidder << " --> " << matcher.leftMatch(bidder) << endl;
//    return 0;
//  }

  ptime time_end(second_clock::local_time());
  time_duration duration(time_end - time_start);
  cout << "Duration: " << duration << endl;

  return 0;
}
