#include <fastlib/fastlib.h>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "anmf.h"
#include "kdnode.h"

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
 
  Matrix query;
  data::Load(vm["query"].as<string>().c_str(), &query);
  anmf::KDNode *qRoot = new anmf::KDNode(query);

  ptime time_end(second_clock::local_time());
  time_duration duration(time_end - time_start);
  cout << "Duration: " << duration << endl;

  return 0;
}
