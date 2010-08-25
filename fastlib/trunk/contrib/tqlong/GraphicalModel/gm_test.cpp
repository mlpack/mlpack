#include <iostream>
#include <fastlib/fastlib.h>
#include "gm.h"
using namespace std;

void testNaiveInference();

int main(int argc, char** argv)
{
//  fx_module* root = fx_init(argc, argv, NULL);
  testNaiveInference();
//  fx_done(root);
}

template <typename Inference, typename Variable>
void printBelief(const typename Inference::vertex_type& u,
                 const typename Inference::belief_type& blf)
{
  if (u->isVariable())
  {
    const Variable* var = (const Variable*) u->variable();
    cout << var->name() << " belief: ";
    BOOST_FOREACH(const typename Inference::belief_type::value_type& p, blf)
  //  for (typename Inference::belief_type::iterator it = blf.begin(); it != blf.end(); it++)
    {
      cout << (var->valueMap()->getForward(FINITE_VALUE(p.first))) << " = " << p.second << " ";
    }
    cout << endl;
  }
  else // the average of this factor is blf[0]
  {
    const typename Inference::factor_type* f = (const typename Inference::factor_type*) u->factor();
    cout << f->toString() << " average = " << blf.get(0) << endl;
  }
  // cout << "equal = " << (b.size() < 2 ? 0 : b[0] == b[1]) << endl;
  // cout << "greater = " << (b.size() < 2 ? 0 : b[0] > b[1]) << endl;
  // cout << "less = " << (b.size() < 2 ? 0 : b[0] < b[1]) << endl;
}

void testNaiveInference()
{
  typedef gm::ConvergenceMeasure Cvm;
  typedef gm::FiniteVar<std::string> Variable;
  typedef Variable::int_value_map_type value_map_type;

  typedef gm::Assignment Assignment;
  typedef gm::Logarithm Logarithm;
  typedef gm::TableF<Logarithm> Factor;
  typedef gm::FactorGraph<Factor> Graph;
  typedef gm::MessagePriorityInference<Factor> Inference;
  typedef Inference::belief_type belief_type;
  typedef Inference::belief_map_type belief_map_type;
//  void printBelief<Inference>(belief_type blf);

  struct GraphBuilder
  {
    GraphBuilder(gm::Variable* rain, gm::Variable* sprinklet, gm::Variable* wet,
                 const gm::Assignment& evidence, Graph& fg)
    {
      double w1[2][2] = {{0, -0.5},{-2,0.5}};
      Factor f1("rw", gm::Domain() << rain << wet);
      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
        Assignment a;
        a[rain] = i;
        a[wet] = j;
        f1[a] = Logarithm(w1[i][j],1);
      }

      double w2[2][2] = {{0, -0.5},{-1,0}};
      Factor f2("sw", gm::Domain() << sprinklet << wet);
      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
        Assignment a;
        a[sprinklet] = i;
        a[wet] = j;
        f2[a] = Logarithm(w2[i][j],1);
      }
      Factor f1_res(f1, evidence);
      Factor f2_res(f2, evidence);
      fg.add(f1_res);
      fg.add(f2_res);
    }
  };

  gm::Universe u;

  value_map_type vMap;
  vMap << value_map_type::pair_type(0, "FALSE") << value_map_type::pair_type(1, "TRUE");

  gm::Variable* rain = u.newVariable("rain", Variable("temp", vMap));
  gm::Variable* sprinklet = u.newVariable("sprinklet", Variable("temp", vMap));
  gm::Variable* wet = u.newVariable("wet", Variable("temp", vMap));

  cout << u.toString("Universe RSW") << endl;

  Assignment e;
//  e[rain] = 0;
//  e[sprinklet] = 1;
  e[wet] = 0;
  cout << "Evidence = " << e.toString() << endl;

  Graph fg("RSW");
  GraphBuilder(rain, sprinklet, wet, e, fg);
  cout << fg.toString() << endl;

//   Inference::_Base::_Base bp(fg);               // NaiveInference
//  Inference::_Base bp(fg);                      // SumProductInference cvm = Cvm(Cvm::Iter)
  Inference bp(fg, Cvm(Cvm::Iter|Cvm::Change)); // MessagePriorityInference
  bp.run();

  cout << "---------------------- Inference result ----------------------" << endl;
  belief_map_type beliefs = bp.beliefs();
  BOOST_FOREACH (const belief_map_type::value_type& p, beliefs)
  {
    printBelief<Inference, Variable>(p.first, p.second);
  }
}
