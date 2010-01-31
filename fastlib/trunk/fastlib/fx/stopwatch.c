// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * @file stopwatch.c
 *
 * Definitions for timer utilities.
 */

#include "stopwatch.h"

#include "fastlib/base/debug.h"

#include <unistd.h>
#include <time.h>
#include <sys/time.h>



/* TODO: Problem with user time? */

void timestamp_init(struct timestamp *dest)
{
  memset(dest, 0, sizeof(struct timestamp));
}

void timestamp_add(struct timestamp *dest, const struct timestamp *src)
{
  dest->micros += src->micros;
#ifdef HAVE_RDTSC
  dest->cycles += src->cycles;
#endif
  dest->cpu.tms_utime += src->cpu.tms_utime;
  dest->cpu.tms_stime += src->cpu.tms_stime;
  dest->cpu.tms_cutime += src->cpu.tms_cutime;
  dest->cpu.tms_cstime += src->cpu.tms_cstime;
}

void timestamp_sub(struct timestamp *dest, const struct timestamp *src)
{
  dest->micros -= src->micros;
#ifdef HAVE_RDTSC
  dest->cycles -= src->cycles;
#endif
  dest->cpu.tms_utime -= src->cpu.tms_utime;
  dest->cpu.tms_stime -= src->cpu.tms_stime;
  dest->cpu.tms_cutime -= src->cpu.tms_cutime;
  dest->cpu.tms_cstime -= src->cpu.tms_cstime;
}

void timestamp_now(struct timestamp *dest)
{
  struct timeval tv;

  /* Highest precision first */
#ifdef HAVE_RDTSC
  RDTSC(dest->cycles);
#endif
  gettimeofday(&tv, NULL);
  dest->micros = 1000000 * (tsc_t)tv.tv_sec + tv.tv_usec;
  times(&dest->cpu);
}

void timestamp_now_rev(struct timestamp *dest)
{
  struct timeval tv;

  /* Highest precision last */
  times(&dest->cpu);
  gettimeofday(&tv, NULL);
  dest->micros = 1000000 * (tsc_t)tv.tv_sec + tv.tv_usec;
#ifdef HAVE_RDTSC
  RDTSC(dest->cycles);
#endif
}



void stopwatch_init(struct stopwatch *timer)
{
  timestamp_init(&timer->total);
  timestamp_init(&timer->start);
}

void stopwatch_start(struct stopwatch *timer)
{
  DEBUG_WARN_MSG_IF(STOPWATCH_ACTIVE(timer),
      "Restarting active stopwatch.");

  timestamp_now_rev(&timer->start);
}

void stopwatch_stop(struct stopwatch *timer, const struct timestamp *now)
{
  if (likely(STOPWATCH_ACTIVE(timer))) {
    timestamp_add(&timer->total, now);
    timestamp_sub(&timer->total, &timer->start);
    timestamp_init(&timer->start);
  } else {
    DEBUG_ONLY(NONFATAL("Stopping inactive stopwatch."));
  }
}
