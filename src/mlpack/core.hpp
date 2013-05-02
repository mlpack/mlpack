/***
 * @file core.hpp
 *
 * Include all of the base components required to write MLPACK methods, and the
 * main MLPACK Doxygen documentation.
 *
 * This file is part of MLPACK 1.0.5.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_HPP
#define __MLPACK_CORE_HPP

/**
 * @mainpage MLPACK Documentation
 *
 * @section intro_sec Introduction
 *
 * MLPACK is an intuitive, fast, scalable C++ machine learning library, meant to
 * be a machine learning analog to LAPACK.  It aims to implement a wide array of
 * machine learning methods and function as a "swiss army knife" for machine
 * learning researchers.  The MLPACK development website can be found at
 * http://mlpack.org.
 *
 * MLPACK uses the Armadillo C++ matrix library (http://arma.sourceforge.net)
 * for general matrix, vector, and linear algebra support.  MLPACK also uses the
 * program_options, math_c99, and unit_test_framework components of the Boost
 * library; in addition, LibXml2 is used.
 *
 * @section howto How To Use This Documentation
 *
 * This documentation is API documentation similar to Javadoc.  It isn't
 * necessarily a tutorial, but it does provide detailed documentation on every
 * namespace, method, and class.
 *
 * Each MLPACK namespace generally refers to one machine learning method, so
 * browsing the list of namespaces provides some insight as to the breadth of
 * the methods contained in the library.
 *
 * To generate this documentation in your own local copy of MLPACK, you can
 * simply use Doxygen, from the root directory (@c /mlpack/trunk/ ):
 *
 * @code
 * $ doxygen
 * @endcode
 *
 * @section executables Executables
 *
 * MLPACK provides several executables so that MLPACK methods can be used
 * without any need for knowledge of C++.  These executables are all
 * self-documented, and that documentation can be accessed by running the
 * executables with the '-h' or '--help' flag.
 *
 * A full list of executables is given below:
 *
 * allkfn, allknn, emst, gmm, hmm_train, hmm_loglik, hmm_viterbi, hmm_generate,
 * kernel_pca, kmeans, lars, linear_regression, local_coordinate_coding, mvu,
 * nbc, nca, pca, radical, sparse_coding
 *
 * @section tutorial Tutorials
 *
 * A few short tutorials on how to use MLPACK are given below.
 *
 *  - @ref build
 *  - @ref matrices
 *  - @ref iodoc
 *  - @ref timer
 *  - @ref sample
 *
 * Tutorials on specific methods are also available.
 *
 *  - @ref nstutorial
 *  - @ref lrtutorial
 *  - @ref rstutorial
 *  - @ref dettutorial
 *  - @ref emst_tutorial
 *  - @ref kmtutorial
 *  - @ref fmkstutorial
 *
 * @section methods Methods in MLPACK
 *
 * The following methods are included in MLPACK:
 *
 *  - Density Estimation Trees - mlpack::det::DTree
 *  - Euclidean Minimum Spanning Trees - mlpack::emst::DualTreeBoruvka
 *  - Gaussian Mixture Models (GMMs) - mlpack::gmm::GMM
 *  - Hidden Markov Models (HMMs) - mlpack::hmm::HMM
 *  - Kernel PCA - mlpack::kpca::KernelPCA
 *  - K-Means Clustering - mlpack::kmeans::KMeans
 *  - Least-Angle Regression (LARS/LASSO) - mlpack::regression::LARS
 *  - Local Coordinate Coding - mlpack::lcc::LocalCoordinateCoding
 *  - Locality-Sensitive Hashing - mlpack::neighbor::LSHSearch
 *  - Naive Bayes Classifier - mlpack::naive_bayes::NaiveBayesClassifier
 *  - Neighborhood Components Analysis (NCA) - mlpack::nca::NCA
 *  - Principal Components Analysis (PCA) - mlpack::pca::PCA
 *  - RADICAL (ICA) - mlpack::radical::Radical
 *  - Simple Least-Squares Linear Regression -
 *        mlpack::regression::LinearRegression
 *  - Sparse Coding - mlpack::sparse_coding::SparseCoding
 *  - Tree-based neighbor search (AllkNN, AllkFN) -
 *        mlpack::neighbor::NeighborSearch
 *  - Tree-based range search - mlpack::range::RangeSearch
 *
 * @section remarks Final Remarks
 *
 * This software was written in the FASTLab (http://www.fast-lab.org), which is
 * in the School of Computational Science and Engineering at the Georgia
 * Institute of Technology.
 *
 * MLPACK contributors include:
 *
 *   - Ryan Curtin <gth671b@mail.gatech.edu>
 *   - James Cline <james.cline@gatech.edu>
 *   - Neil Slagle <nslagle3@gatech.edu>
 *   - Matthew Amidon <mamidon@gatech.edu>
 *   - Vlad Grantcharov <vlad321@gatech.edu>
 *   - Ajinkya Kale <kaleajinkya@gmail.com>
 *   - Bill March <march@gatech.edu>
 *   - Dongryeol Lee <dongryel@cc.gatech.edu>
 *   - Nishant Mehta <niche@cc.gatech.edu>
 *   - Parikshit Ram <p.ram@gatech.edu>
 *   - Rajendran Mohan <rmohan88@gatech.edu>
 *   - Trironk Kiatkungwanglai <trironk@gmail.com>
 *   - Patrick Mason <patrick.s.mason@gmail.com>
 *   - Chip Mappus <cmappus@gatech.edu>
 *   - Hua Ouyang <houyang@gatech.edu>
 *   - Long Quoc Tran <tqlong@gmail.com>
 *   - Noah Kauffman <notoriousnoah@gmail.com>
 *   - Guillermo Colon <gcolon7@mail.gatech.edu>
 *   - Wei Guan <wguan@cc.gatech.edu>
 *   - Ryan Riegel <rriegel@cc.gatech.edu>
 *   - Nikolaos Vasiloglou <nvasil@ieee.org>
 *   - Garry Boyer <garryb@gmail.com>
 *   - Andreas Löf <andreas.lof@cs.waikato.ac.nz>
 */

// First, standard includes.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <iostream>

// Defining _USE_MATH_DEFINES should set M_PI.
#define _USE_MATH_DEFINES
#include <math.h>

// For tgamma().
#include <boost/math/special_functions/gamma.hpp>

// But if it's not defined, we'll do it.
#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279
#endif

// Clean up unfortunate Windows preprocessor definitions.
// Use std::min and std::max!
#ifdef _WIN32
  #ifdef min
    #undef min
  #endif

  #ifdef max
    #undef max
  #endif
#endif

// Give ourselves a nice way to force functions to be inline if we need.
#define force_inline
#if defined(__GNUG__) && !defined(DEBUG)
  #undef force_inline
  #define force_inline __attribute__((always_inline))
#elif defined(_MSC_VER)
  #undef force_inline && !defined(DEBUG)
  #define force_inline __forceinline
#endif

// Now MLPACK-specific includes.
#include <mlpack/core/arma_extend/arma_extend.hpp> // Includes Armadillo.
#include <mlpack/core/util/log.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>
#include <mlpack/core/math/clamp.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/math/lin_alg.hpp>
#include <mlpack/core/math/range.hpp>
#include <mlpack/core/math/round.hpp>
#include <mlpack/core/util/save_restore_utility.hpp>
#include <mlpack/core/dists/discrete_distribution.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>

#endif
