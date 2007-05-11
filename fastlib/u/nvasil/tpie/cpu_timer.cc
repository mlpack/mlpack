// Copyright (c) 1995 Darren Vengroff
//
// File: cpu_timer.cpp
// Author: Darren Vengroff <darrenv@eecs.umich.edu>
// Created: 1/11/95
//

#include <versions.h>
VERSION(cpu_timer_cpp,"$Id: cpu_timer.cpp,v 1.9 2004/08/17 16:48:50 jan Exp $");

#include <cpu_timer.h>

cpu_timer::cpu_timer() :
        running(false)
{
    TPIE_OS_SET_CLOCK_TICK;

    elapsed_real = 0;
}

cpu_timer::~cpu_timer()
{
}

void cpu_timer::sync()
{
    clock_t current_real;

    TPIE_OS_TMS current;
    TPIE_OS_SET_CURRENT_TIME(current);
    TPIE_OS_UNIX_ONLY_SET_ELAPSED_TIME(current);

    elapsed_real += current_real - last_sync_real;

    last_sync = current;
    last_sync_real = current_real;
}


void cpu_timer::start()
{
    if (!running) {
	TPIE_OS_LAST_SYNC_REAL_DECLARATION;
	running = true;
    }
}

void cpu_timer::stop()
{
    if (running) {
        sync();
        running = false;
    }
}

void cpu_timer::reset()
{
    if (running) {		
	TPIE_OS_LAST_SYNC_REAL_DECLARATION;
    }
    
    TPIE_OS_SET_CLOCK_TICK;	
    elapsed_real = 0;
}

double cpu_timer::user_time() {
  if (running) sync();
  TPIE_OS_USER_TIME_BODY;
}

double cpu_timer::system_time() {
  if (running) sync();
  TPIE_OS_USER_TIME_BODY;
}

double cpu_timer::wall_time() {
  if (running) sync();
  return double(elapsed_real) / double(clock_tick);
}

ostream &operator<<(ostream &s, cpu_timer &wt)
{
    if (wt.running) {
        wt.sync();
    }
    
    TPIE_OS_OPERATOR_OVERLOAD;
}


