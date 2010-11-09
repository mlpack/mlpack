#include <fastlib/fastlib.h>
#include <limits>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/foreach.hpp>
#include "matching.h"
#include "kdnode.h"
#include "single_tree.h"

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

MATCHING_NAMESPACE_BEGIN;

typedef double PointStats;
class NodeStats
{
public:
  double minPrice_;
  Vector minBox_, maxBox_;

  std::string toString() const
  {
    std::stringstream s;
    s << "minPrice = " << minPrice_ << " minBox = " << match::toString(minBox_) << " maxBox = " << match::toString(maxBox_);
    return s.str();
  }
};

template <>
    void KDNodeStats<PointStats, NodeStats>::setLeafStats(bool init)
{
  nodeStats_.minPrice_ = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n_points(); i++)
  {
    double val = pointStats(i);
    if (nodeStats_.minPrice_ > val) nodeStats_.minPrice_ = val;
  }
  if (init)
  {
    nodeStats_.minBox_.Init(n_dim());
    nodeStats_.maxBox_.Init(n_dim());
    nodeStats_.minBox_.SetAll(std::numeric_limits<double>::infinity());
    nodeStats_.maxBox_.SetAll(-std::numeric_limits<double>::infinity());
    for (int dim = 0; dim < n_dim(); dim++)
    {
      for (int i = 0; i < n_points(); i++)
      {
        double val = get(dim, i);
        if (nodeStats_.minBox_[dim] > val) nodeStats_.minBox_[dim] = val;
        if (nodeStats_.maxBox_[dim] < val) nodeStats_.maxBox_[dim] = val;
      }
    }
  }
}

template <>
    void KDNodeStats<PointStats, NodeStats>::setNonLeafStats(bool init)
{
  nodeStats_.minPrice_ = std::numeric_limits<double>::infinity();
  if (init)
  {
    nodeStats_.minBox_.Init(n_dim());
    nodeStats_.maxBox_.Init(n_dim());
    nodeStats_.minBox_.SetAll(std::numeric_limits<double>::infinity());
    nodeStats_.maxBox_.SetAll(-std::numeric_limits<double>::infinity());
  }
  BOOST_FOREACH(KDNode* child, children_)
  {
    const NodeStats& childStats = ((KDNodeStats*) child)->nodeStats();
    if (childStats.minPrice_ < nodeStats_.minPrice_) nodeStats_.minPrice_ = childStats.minPrice_;
    if (init) for (int dim = 0; dim < n_dim(); dim++)
    {
      if (childStats.minBox_[dim] < nodeStats_.minBox_[dim]) nodeStats_.minBox_[dim] = childStats.minBox_[dim];
      if (childStats.maxBox_[dim] > nodeStats_.maxBox_[dim]) nodeStats_.maxBox_[dim] = childStats.maxBox_[dim];
    }
  }
}

template <>
    std::string KDNodeStats<PointStats, NodeStats>::toString(int depth) const
{
  std::stringstream s;
  if (depth == 0)
  {
    for (int i = 0; i < n_points_; i++)
    {
      Vector p_i;
      getPoint(i, p_i);
      s << i << " --> " << match::toString(p_i) << "\n";
    }
  }
  for (int i = 0; i < depth; i++) s << "  ";
  s << "-- Node (" << dfsIndex_;
  for (int i = 1; i < n_points_; i++)
    s << "," << dfsIndex_ + i;
  s << ")\n";
  for (int i = 0; i < depth+1; i++) s << "  ";
  s << " " << nodeStats_.toString() << "\n";
  for (int i = 0; i < n_children(); i++)
    s << ((KDNodeStats*) child(i))->toString(depth+1);
  return s.str();
}

template <>
    double SingleTree<Vector, KDNodeStats<PointStats, NodeStats> >::distance(const point_type &q, node_type &ref, int index)
{
  Vector r;
  ref.getPoint(index, r);
  return sqrt(la::DistanceSqEuclidean(q, r))+ref.pointStats(index);
}

template <>
    double SingleTree<Vector, KDNodeStats<PointStats, NodeStats> >::distance(const point_type &q, node_type &ref)
{
  const NodeStats& stats = ref.nodeStats();
  double s = 0;
  for (int dim = 0; dim < ref.n_dim(); dim++)
  {
    if (q[dim] < stats.minBox_[dim]) s += math::Sqr(q[dim] - stats.minBox_[dim]);
    else if (q[dim] > stats.maxBox_[dim]) s += math::Sqr(q[dim] - stats.maxBox_[dim]);
  }
  return sqrt(s)+stats.minPrice_;
}

MATCHING_NAMESPACE_END;

int main(int argc, char** argv)
{
  process_options(argc, argv);

  if (vm["random"].as<int>() > 0)
  {
    generateRandom(vm["reference"].as<string>().c_str(), vm["query"].as<string>().c_str());
  }

  ptime time_start(second_clock::local_time());
 
  Matrix reference, query;
  data::Load(vm["reference"].as<string>().c_str(), &reference);
  data::Load(vm["query"].as<string>().c_str(), &query);

  typedef match::KDNodeStats<match::PointStats, match::NodeStats> Node;
  Node *rRoot = new Node(reference);
  rRoot->split();
  for (int i = 0; i < rRoot->n_points(); i++)
    rRoot->setPointStats(i, 0);
  cout << "Done set stats\n";
  rRoot->visit(true);
  cout << "Done visit\n";
  cout << rRoot->toString() << "\n";

  typedef match::SingleTree<Vector, Node> SingleTree;
  SingleTree algo;

  for (int i = 0; i < query.n_cols(); i++)
  {
    int minIndex = -1; double minDistance = std::numeric_limits<double>::infinity();
    Vector q, r;
    query.MakeColumnVector(i, &q);
    double pruned = algo.nearestNeighbor(q, *rRoot, minIndex, minDistance);
    rRoot->getPoint(minIndex, r);
    cout << "q = " << match::toString(q) << " --> r = " << match::toString(r)
         << " index = " << minIndex << " dist = " << minDistance
         << " pruned = " << pruned << "\n";
  }

  delete rRoot;

  ptime time_end(second_clock::local_time());
  time_duration duration(time_end - time_start);
  cout << "Duration: " << duration << endl;

  return 0;
}
