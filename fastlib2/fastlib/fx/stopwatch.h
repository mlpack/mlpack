// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * @file stopwatch.h
 *
 * Timer utilities used by FASTexec; there are convenient methods in
 * fx.h to make use of these timers.  Also, see the RDTSC macro for a
 * minimal-impact cycle counter.
 */

#ifndef FX_TIMER_H
#define FX_TIMER_H

#include "fastlib/base/common.h"

#include <sys/times.h>

EXTERN_C_BEGIN

/** Cycle counter data type. */
typedef uint64 tsc_t;
/** Length modifier for emitting tsc_t with printf. */
#define LTSC L64

/* TODO: Check x86_64 and other architectures */

#if defined(__i386__)
/** Read the number of cycles executed into a tsc_t variable. */
#define RDTSC(tscv) __asm__ volatile (".byte 0x0f, 0x31" : "=A" (tscv))
#define HAVE_RDTSC
#endif

/** Snapshot of both CPU and real time. */
struct timestamp {
  /** Microseconds since an unknown epoch. */
  tsc_t micros;
#ifdef HAVE_RDTSC
  /** CPU cycles since an unknown epoch. */
  tsc_t cycles;
#endif
  /** CPU time as returned by times(2). */
  struct tms cpu;
};

/** Main timer structure. */
struct stopwatch {
  /** Total time elapsed in all previous start/stop runs. */
  struct timestamp total;
  /** The most recent start time. */
  struct timestamp start;
};

/** Initializes a timestamp to zero. */
void timestamp_init(struct timestamp *snapshot);
/** Element-wise addition of a timestamp. */
void timestamp_add(struct timestamp *dest, const struct timestamp *src);
/** Element-wise subtraction of a timestamp. */
void timestamp_sub(struct timestamp *dest, const struct timestamp *src);
/**
 * Records the current time.
 *
 * The highest-precision operations are found first.
 */
void timestamp_now(struct timestamp *dest);
/**
 * Records the current time.
 *
 * The highest-precision operations are found last.
 */
void timestamp_now_rev(struct timestamp *dest);

/** Initializes a timer. */
void stopwatch_init(struct stopwatch *timer);
/** Starts a timing run. */
void stopwatch_start(struct stopwatch *timer);
/** Stops a timing run, accumulating it into the total. */
void stopwatch_stop(struct stopwatch *timer, const struct timestamp *now);

/** Test whether a timer is active. */
#define STOPWATCH_ACTIVE(timer) ((timer)->start.micros != 0)

EXTERN_C_END

#endif
