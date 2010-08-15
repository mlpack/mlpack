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
    void printBelief(typename Inference::belief_type blf, Variable* var)
{
  for (typename Inference::belief_type::iterator it = blf.begin(); it != blf.end(); it++)
  {
    const gm::Value& val = (*it).first;
    cout << (var->valueMap()->getForward(FINITE_VALUE(val))) << " = (" << (*it).second << ") ";
  }
  cout << endl;
  // cout << "equal = " << (b.size() < 2 ? 0 : b[0] == b[1]) << endl;
  // cout << "greater = " << (b.size() < 2 ? 0 : b[0] > b[1]) << endl;
  // cout << "less = " << (b.size() < 2 ? 0 : b[0] < b[1]) << endl;
}

void testNaiveInference()
{
  typedef gm::FiniteVar<std::string> Variable;
  typedef Variable::int_value_map_type value_map_type;

  typedef gm::Assignment Assignment;
  typedef gm::Logarithm Logarithm;
  typedef gm::TableF<Logarithm> Factor;
  typedef gm::FactorGraph<Factor> Graph;
  typedef gm::NaiveInference<Factor> Inference;
  typedef Inference::belief_type belief_type;
  typedef Inference::belief_map_type belief_map_type;
//  void printBelief<Inference>(belief_type blf);

  gm::Universe u;

  value_map_type vMap;
  vMap << value_map_type::pair_type(0, "FALSE") << value_map_type::pair_type(1, "TRUE");

  gm::Variable* rain = u.newVariable("rain", Variable("temp", vMap));
  gm::Variable* sprinklet = u.newVariable("sprinklet", Variable("temp", vMap));
  gm::Variable* wet = u.newVariable("wet", Variable("temp", vMap));

  u.print("universe");

  double w1[2][2] = {{0, -0.5},{-2,0}};
  Factor f1(gm::Domain() << rain << wet);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
    {
    Assignment a;
    a[rain] = i;
    a[wet] = j;
    f1[a] = Logarithm(w1[i][j],1);
  }

  double w2[2][2] = {{0, -0.5},{-1,0}};
  Factor f2(gm::Domain() << sprinklet << wet);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
    {
    Assignment a;
    a[sprinklet] = i;
    a[wet] = j;
    f2[a] = Logarithm(w2[i][j],1);
  }

  Assignment e;
  e[rain] = 1;
  e[sprinklet] = 1;
  f1.restricted(e);
  f2.restricted(e);

  Graph fg;
  fg.add(f1);
  fg.add(f2);
  fg.print();

  Inference bp(fg);

  bp.run();

  belief_map_type beliefs = bp.beliefs();
  for (belief_map_type::iterator it = beliefs.begin(); it != beliefs.end(); it++)
  {
    cout << (*it).first->name() << " belief: ";
    printBelief<Inference, Variable>((*it).second, (Variable*) (it->first));
  }
}

/* TRASH
  gm::Logarithm a(0), b(0, 1);

  cout << "a = " << a << " = " << a.originalValue() << endl
       << "b = " << b << " = " << b.originalValue() << endl;

  gm::Logarithm c(0), d(0, 1);

  c = a; d = d+b; d += d;
  cout << "c = " << c << " = " << c.originalValue() << endl
       << "d = " << d << " = " << d.originalValue() << endl;

  gm::Logarithm e(0), f(0, 1);
  e = d - b; f = d; f -= d;
  cout << "e = " << e << " = " << e.originalValue() << endl
       << "f = " << f << " = " << f.originalValue() << endl;

  gm::Logarithm g, h;
  g = d * b; h = g; h *= g;
  cout << "g = " << g << " = " << g.originalValue() << endl
       << "h = " << h << " = " << h.originalValue() << endl;
  
  gm::Logarithm i, j;
  i = h / g; j = h; j /= g;
  cout << "i = " << i << " = " << i.originalValue() << endl
       << "j = " << j << " = " << j.originalValue() << endl;
*/

/* TRASH
typedef gm::NaiveInference<gm::TableF<double> >::belief_type belief_type;
typedef gm::NaiveInference<gm::TableF<double> >::belief_map_type belief_map_type;

  gm::Universe u;
  gm::FiniteVariable* v = u.newFiniteVariable("V", 2);
  gm::FiniteVariable* v1 = u.newFiniteVariable("V1", *v);
  gm::FiniteVariable* v2 = u.newFiniteVariable("V2", *v);
  u.print("universe");

  gm::Domain dom;
  dom << v << v1;

  gm::Assignment eres;
//  eres[v] = 0;
//  eres[v1] = 0;
  gm::TableF<double> f(dom, eres);

  gm::Assignment e[2][2];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
    {
      e[i][j][v] = i;
      e[i][j][v1] = j;
      DEBUG_ASSERT(e[i][j].checkFiniteValueIntegrity());
      f[e[i][j]] = i + j;
    }

  //cout << "factor = " << endl;
  //f.print();

//  f.restricted(e[0][0]);
//  cout << "AFTER RESTRICTED" << endl;
//  f.print();
  gm::TableF<double> f1(gm::Domain() << v);
  gm::Assignment e1;
  for (int i = 0; i < 2; i++)
  {
    e1[v] = i;
    f1[e1] = i+1;
  }
  
  gm::TableF<double> f2(gm::Domain() << v2);
  gm::Assignment e2;
  for (int i = 0; i < 2; i++)
  {
    e2[v2] = i;
    f2[e2] = i+1;
  }
  
  gm::TableF<double> f3(gm::Domain() << v2);
  gm::Assignment e3;
  for (int i = 0; i < 2; i++)
  {
    e3[v2] = i;
    f3[e3] = 2-i;
  }
  
  gm::FactorGraph<gm::TableF<double> > fg;
  fg.add(f);
  fg.add(f1);
  fg.add(f2);
  fg.add(f3);
  fg.print();

  gm::NaiveInference<gm::TableF<double> > bp(fg);
  bp.run();

  belief_map_type beliefs = bp.beliefs();
  for (belief_map_type::iterator it = beliefs.begin(); it != beliefs.end(); it++)
  {
    cout << (*it).first->name() << " belief: ";
    printBelief((*it).second);
  }

 */

/* TRASH
     gm::Domain dom;
  dom << v << v1;

  gm::Assignment eres;
//  eres[v] = 0;
//  eres[v1] = 0;
  gm::TableF<Logarithm> f(dom, eres);

  gm::Assignment e[2][2];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
    {
      e[i][j][v] = i;
      e[i][j][v1] = j;
      DEBUG_ASSERT(e[i][j].checkFiniteValueIntegrity());
      f[e[i][j]] = i + j;
    }

  //cout << "factor = " << endl;
  //f.print();

//  f.restricted(e[0][0]);
//  cout << "AFTER RESTRICTED" << endl;
//  f.print();
  gm::TableF<Logarithm> f1(gm::Domain() << v);
  gm::Assignment e1;
  for (int i = 0; i < 2; i++)
  {
    e1[v] = i;
    f1[e1] = i+1;
  }
  
  gm::TableF<Logarithm> f2(gm::Domain() << v2);
  gm::Assignment e2;
  for (int i = 0; i < 2; i++)
  {
    e2[v2] = i;
    f2[e2] = i+1;
  }
  
  gm::TableF<Logarithm> f3(gm::Domain() << v2);
  gm::Assignment e3;
  for (int i = 0; i < 2; i++)
  {
    e3[v2] = i;
    f3[e3] = 2-i;
  }
  
  gm::FactorGraph<gm::TableF<Logarithm> > fg;
  fg.add(f);
  fg.add(f1);
  fg.add(f2);
  fg.add(f3);
  fg.print();

  gm::NaiveInference<gm::TableF<Logarithm> > bp(fg);
  bp.run();

  belief_map_type beliefs = bp.beliefs();
  for (belief_map_type::iterator it = beliefs.begin(); it != beliefs.end(); it++)
  {
    cout << (*it).first->name() << " belief: ";
    printBelief((*it).second);
  }
*/
