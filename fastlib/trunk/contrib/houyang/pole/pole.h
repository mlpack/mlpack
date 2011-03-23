#ifndef POLE_H
#define POLE_H

#include <cstring>
#include <map>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "learner.h"
#include "kernel.h"
#include "transform.h"
#include "opt_ogd.h"
#include "opt_ogd_k.h"
#include "opt_ogd_t.h"
#include "opt_oeg.h"
#include "opt_wm.h"

using namespace boost::posix_time;

class Pole {
 public:
  map<string, string> m_;
  Learner *L_;
  bool    batch_; // online or batch learning
  string  opt_name_; // learning/optimization method
  
 public:
  Pole();
  ~Pole();
  void ParseArgs(int argc, char *argv[]);
  void Run();

 private:
  void ArgsSanityCheck();
};

#endif
