/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file cc.cc
 *
 * Implementations for bare-necessities FASTlib programming in C++.
 */

#include "cc.h"
#include "debug.h"
//#include "cc.h"
//#include "debug.h"

const double DBL_NAN = std::numeric_limits<double>::quiet_NaN();
const float FLT_NAN = std::numeric_limits<float>::quiet_NaN();
const double DBL_INF = std::numeric_limits<double>::infinity();
const float FLT_INF = std::numeric_limits<float>::infinity();

#if defined(DEBUG) || defined(PROFILE)

namespace cc__private {
  /** Hidden class that emits messages for debug and profile modes. */
  class InformDebug {
   public:
    InformDebug() {
      PROFILE_ONLY(NOTIFY_STAR("Profiling information available with:\n"));
      PROFILE_ONLY(NOTIFY_STAR("  gprof $THIS > prof.out && less prof.out\n"));
      DEBUG_ONLY(NOTIFY_STAR(
          ANSI_BLACK"Program compiled with debug checks."ANSI_CLEAR"\n"));
    }
    ~InformDebug() {
      PROFILE_ONLY(NOTIFY_STAR("Profiling information available with:\n"));
      PROFILE_ONLY(NOTIFY_STAR("  gprof $THIS > prof.out && less prof.out\n"));
      DEBUG_ONLY(NOTIFY_STAR(
          ANSI_BLACK"Program compiled with debug checks."ANSI_CLEAR"\n"));
    }
  };

  /** Global instance prints messages before and after computation. */
  InformDebug inform_debug;
};

#endif
