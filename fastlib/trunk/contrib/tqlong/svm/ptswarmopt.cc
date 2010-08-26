#include <fastlib/fastlib.h>
#include "svm.h"

using namespace SVMLib;

namespace SVMLib {

struct Particle {
  Vector alpha; 
  IndexSet SVs;
  double offset;

  double objective;

  Vector v_alpha;
  double v_offset; // velocity v_alpha & v_offset

  Particle(index_t n) : SVs(n) {
    alpha.Init(n);
    v_alpha.Init(n);
  }
  Particle(index_t n, const Vector& y, const Vector& box, double sv_rate) 
    : SVs(n) {
    alpha.Init(n); alpha.SetZero();
    double s = 0;
    for (index_t i = 0; i < n; i++)
      if (math::Random() < sv_rate) {
	alpha[i] = math::Random()*box[i];
	s += alpha[i]*y[i];
	SVs.addremove(i, true);
      }
    // make alpha orthogonal to y
    for (index_t ii = 0; ii < SVs.get_n(); ii++) {
      //index_t i = SVs[ii];
      //TODO alpha[i]
    }

    offset = math::Random()*100-100;
    v_alpha.Init(n);
    for (index_t i = 0; i < n; i++)
      v_alpha[i] = math::Random(-box[i]/10,box[i]/10);
    // make v orthogonal to y
    la::AddExpert(-la::Dot(v_alpha, y)/n, y, &v_alpha);
    v_offset = math::Random()*10-10;
  }

  void cal_objective(Kernel& kernel, const Vector& y) {
    this->objective = (double) svm_total_error(kernel, y, alpha, SVs, offset);
  }

  void operator=(const Particle& p) {
    this->alpha.CopyValues(p.alpha);
    this->offset = p.offset;
    this->objective = p.objective;
    this->v_alpha.CopyValues(p.alpha);
    this->v_offset = p.offset;
    this->SVs = p.SVs;
  }
};

typedef ArrayList<Particle> Swarm;

void updateSwarm(Swarm& swarm);
void getBest(Swarm& swarm, Particle& best);

int ptswarmopt(const Matrix& X, const Vector& y, const Vector& box,
	       Kernel& kernel, /* PSOOptions options, */
	       Vector& alpha, IndexSet& SVs, double& offset) {
  index_t n_particle = 10;
  index_t n_points = y.length();
  double sv_rate = 0.1;
  Swarm swarm;

  swarm.Init(); 
  for (index_t i = 0; i < n_particle; i++) {
    Particle temp(n_points, y, box, sv_rate);
    temp.cal_objective(kernel, y);
    swarm.PushBackCopy(temp);
  }

  Particle best(n_points);

  best.objective = INFINITY; getBest(swarm, best);

  for (index_t iter = 0; iter < 10; iter++) {
    updateSwarm(swarm);
    getBest(swarm, best);
  }

  return 0;
}

void updateSwarm(Swarm& swarm) {
  
}

void getBest(Swarm& swarm, Particle& best) {
  index_t min_i = 0;
  for (index_t i = 0; i < swarm.size(); i++) 
    if (swarm[i].objective < swarm[min_i].objective) min_i = i;
  if (swarm[min_i].objective < best.objective)
    best = swarm[min_i];
}

}
