//
// File: tpie_log.cpp
// Authors: Darren Erik Vengroff <dev@cs.duke.edu>
//          Octavian Procopiuc <tavi@cs.duke.edu>
// Created: 5/12/94
//

#include <versions.h>
VERSION(tpie_log_cpp,"$Id: tpie_log.cpp,v 1.16 2004/08/12 12:53:43 jan Exp $");

// We are logging
#define TPL_LOGGING	1

#include <stdlib.h>
#include <time.h>
#include <tpie_tempnam.h>
#include <tpie_log.h>

#define TPLOGPFX "tpielog"

// Local initialization function. Create a permanent repository for the log
// file name. Should be called only once, by theLogName() below.
static char *__tpie_log_name() {
  static char tln[128];
  TPIE_OS_SRANDOM((unsigned int)TPIE_OS_TIME(NULL));
  strncpy(tln, tpie_tempnam(TPLOGPFX, TPLOGDIR), 124);
  strcat(tln, ".txt");
  return tln;
}

char *tpie_log_name() {
  static char *tln = __tpie_log_name();
  return tln;
}


logstream &tpie_log() {
  static logstream log(tpie_log_name(), TPIE_LOG_DEBUG, TPIE_LOG_DEBUG);
  return log;
}

void tpie_log_init(TPIE_LOG_LEVEL level) {
  TP_LOG_SET_THRESHOLD(level);
}
