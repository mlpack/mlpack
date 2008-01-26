// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file timer.h
 *
 * Timer utilities.
 *
 * More likely than not, you will want to use the convenient methods
 * in fx.h rather than this file.  The most useful part of this is the
 * RDTSC macro, which lets you have a minimal-impact timer.
 */

#ifndef FX_TIMER_H
#define FX_TIMER_H

#include "datastore.h"
#include "base/basic_types.h"

#include <sys/times.h>

EXTERN_C_BEGIN

/** Cycle counter data type. */
typedef uint64 tsc_t;
/** Length modifier for emitting tsc_t with printf. */
#define LT L64

/* TODO: Make this conditional on your architecture */
/* TODO: Does this work for x86_64? */

#if defined(__i386__)
/** Read the number of cycles executed into a tsc_t variable. */
#define RDTSC(tscv) __asm__ volatile (".byte 0x0f, 0x31" : "=A" (tscv))
#define HAVE_RDTSC
#endif

/**
 * Snapshot of both CPU and real time.
 */
struct timestamp {
  /** Microseconds since some unknown epoch. */
  tsc_t micros;
#ifdef HAVE_RDTSC
  /** CPU cycles since an unknown epoch. */
  tsc_t cycles;
#endif
  /** CPU time as returned by times(2). */
  struct tms cpu;
};

/**
 * Stopwatch structure.
 */
struct timer {
  /** Total time elapsed in all previous start/stop runs. */
  struct timestamp total;
  /** The most recent start time. */
  struct timestamp start;
};

/**
 * Initializes a timestamp to zero.
 */
void timestamp_init(struct timestamp *snapshot);
/**
 * Element-wise addition of a timestamp.
 */
void timestamp_add(struct timestamp *dest, const struct timestamp *src);
/**
 * Element-wise subtraction of a timestamp.
 */
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

/**
 * Initializes a timer.
 */
void timer_init(struct timer *timer);
/**
 * Starts a timing run.
 */
void timer_start(struct timer *timer);
/**
 * Stops a timing run and accumulates this current run into the total.
 */
void timer_stop(struct timer *timer, const struct timestamp *now);
/**
 * Write out the cumulative times to a datastore.
 */
void timer_emit_results(struct timer *timer, struct datanode *dest);

/**
 * Test whether a timer is active.
 */
#define TIMER_IS_ACTIVE(timer) ((timer)->start.micros != 0)

EXTERN_C_END

#endif
