/**
 * @file timer.c
 *
 * Definitions for timer utilities.
 */

/* TODO: Fix CPU/User Time */

#include "timer.h"
#include "fx.h"
#include "base/debug.h"

#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void timestamp_init(struct timestamp *snapshot)
{
  memset(snapshot, 0, sizeof(*snapshot));
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

void timer_init(struct timer *timer)
{
  timestamp_init(&timer->total);
  timestamp_init(&timer->start);
}

void timer_start(struct timer *timer)
{
  DEBUG_WARN_MSG_IF(TIMER_IS_ACTIVE(timer),
		    "Restarting active timer");

  timestamp_now_rev(&timer->start);
}

void timer_stop(struct timer *timer, const struct timestamp *now)
{
  if (likely(TIMER_IS_ACTIVE(timer))) {
    timestamp_add(&timer->total, now);
    timestamp_sub(&timer->total, &timer->start);
    timestamp_init(&timer->start);
  } else {
    DEBUG_ONLY(NONFATAL("Tried to stop inactive timer"));
  }
}

void timer_emit_results(struct timer *timer, struct datanode *node)
{
  double clockrate = 1.0 / sysconf(_SC_CLK_TCK);

#ifdef HAVE_RDTSC
  fx_format_result(node, "./wall/cycles", "%"LT"u", timer->total.cycles);
#endif
  fx_format_result(node, "./wall/sec", "%f", timer->total.micros / 1e6);
  fx_format_result(node, "./user", "%f",
		   timer->total.cpu.tms_utime * clockrate);
  fx_format_result(node, "./sys", "%f",
		   timer->total.cpu.tms_stime * clockrate);
}
