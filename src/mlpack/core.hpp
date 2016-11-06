/**
 * @file core.hpp
 *
 * Include all of the base components required to write MLPACK methods, and the
 * main MLPACK Doxygen documentation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_HPP
#define MLPACK_CORE_HPP

/**
 * @mainpage mlpack Documentation
 *
 * @section intro_sec Introduction
 *
 * mlpack is an intuitive, fast, scalable C++ machine learning library, meant to
 * be a machine learning analog to LAPACK.  It aims to implement a wide array of
 * machine learning methods and function as a "swiss army knife" for machine
 * learning researchers.  The mlpack development website can be found at
 * http://mlpack.org.
 *
 * mlpack uses the Armadillo C++ matrix library (http://arma.sourceforge.net)
 * for general matrix, vector, and linear algebra support.  mlpack also uses the
 * program_options, math_c99, and unit_test_framework components of the Boost
 * library, and optionally uses libbfd and libdl to give backtraces when
 * compiled with debugging symbols on some platforms.
 *
 * @section howto How To Use This Documentation
 *
 * This documentation is API documentation similar to Javadoc.  It isn't
 * necessarily a tutorial, but it does provide detailed documentation on every
 * namespace, method, and class.
 *
 * Each mlpack namespace generally refers to one machine learning method, so
 * browsing the list of namespaces provides some insight as to the breadth of
 * the methods contained in the library.
 *
 * To generate this documentation in your own local copy of mlpack, you can
 * simply use Doxygen, from the root directory of the project:
 *
 * @code
 * $ doxygen
 * @endcode
 *
 * @section executables Executables
 *
 * mlpack provides several executables so that mlpack methods can be used
 * without any need for knowledge of C++.  These executables are all
 * self-documented, and that documentation can be accessed by running the
 * executables with the '-h' or '--help' flag.
 *
 * A full list of executables is given below:
 *
 * - mlpack_adaboost
 * - mlpack_approx_kfn
 * - mlpack_cf
 * - mlpack_decision_stump
 * - mlpack_det
 * - mlpack_emst
 * - mlpack_fastmks
 * - mlpack_gmm_train
 * - mlpack_gmm_generate
 * - mlpack_gmm_probability
 * - mlpack_hmm_train
 * - mlpack_hmm_loglik
 * - mlpack_hmm_viterbi
 * - mlpack_hmm_generate
 * - mlpack_hoeffding_tree
 * - mlpack_kernel_pca
 * - mlpack_kfn
 * - mlpack_kmeans
 * - mlpack_knn
 * - mlpack_krann
 * - mlpack_lars
 * - mlpack_linear_regression
 * - mlpack_local_coordinate_coding
 * - mlpack_logistic_regression
 * - mlpack_lsh
 * - mlpack_mean_shift
 * - mlpack_nbc
 * - mlpack_nca
 * - mlpack_pca
 * - mlpack_perceptron
 * - mlpack_radical
 * - mlpack_range_search
 * - mlpack_softmax_regression
 * - mlpack_sparse_coding
 *
 * @section tutorial Tutorials
 *
 * A few short tutorials on how to use mlpack are given below.
 *
 *  - @ref build
 *  - @ref matrices
 *  - @ref iodoc
 *  - @ref timer
 *  - @ref sample
 *  - @ref verinfo
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
 *  - @ref amftutorial
 *
 * @section methods Methods in mlpack
 *
 * The following methods are included in mlpack:
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
 *  - Tree-based neighbor search (KNN, KFN) - mlpack::neighbor::NeighborSearch
 *  - Tree-based range search - mlpack::range::RangeSearch
 *
 * @section remarks Final Remarks
 *
 * mlpack contributors include:
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
 *   - Marcus Edel <marcus.edel@fu-berlin.de>
 *   - Mudit Raj Gupta <mudit.raaj.gupta@gmail.com>
 *   - Sumedh Ghaisas <sumedhghaisas@gmail.com>
 *   - Michael Fox <michaelfox99@gmail.com>
 *   - Ryan Birmingham <birm@gatech.edu>
 *   - Siddharth Agrawal <siddharth.950@gmail.com>
 *   - Saheb Motiani <saheb210692@gmail.com>
 *   - Yash Vadalia <yashdv@gmail.com>
 *   - Abhishek Laddha <laddhaabhishek11@gmail.com>
 *   - Vahab Akbarzadeh <v.akbarzadeh@gmail.com>
 *   - Andrew Wells <andrewmw94@gmail.com>
 *   - Zhihao Lou <lzh1984@gmail.com>
 *   - Udit Saxena <saxena.udit@gmail.com>
 *   - Stephen Tu <tu.stephenl@gmail.com>
 *   - Jaskaran Singh <jaskaranvirdi@gmail.com>
 *   - Shangtong Zhang <zhangshangtong.cpp@icloud.com>
 *   - Hritik Jain <hritik.jain.cse13@itbhu.ac.in>
 *   - Vladimir Glazachev <glazachev.vladimir@gmail.com>
 *   - QiaoAn Chen <kazenoyumechen@gmail.com>
 *   - Janzen Brewer <jahabrewer@gmail.com>
 *   - Trung Dinh <dinhanhtrung@gmail.com>
 *   - Tham Ngap Wei <thamngapwei@gmail.com>
 *   - Grzegorz Krajewski <krajekg@gmail.com>
 *   - Joseph Mariadassou <joe.mariadassou@gmail.com>
 *   - Pavel Zhigulin <pashaworking@gmail.com>
 *   - Andy Fang <AndyFang.DZ@gmail.com>
 *   - Barak Pearlmutter <barak+git@pearlmutter.net>
 *   - Ivari Horm <ivari@risk.ee>
 *   - Dhawal Arora <d.p.arora1@gmail.com>
 *   - Alexander Leinoff <alexander-leinoff@uiowa.edu>
 *   - Palash Ahuja <abhor902@gmail.com>
 *   - Yannis Mentekidis <mentekid@gmail.com>
 *   - Ranjan Mondal <ranjan.rev@gmail.com>
 *   - Mikhail Lozhnikov <lozhnikovma@gmail.com>
 *   - Marcos Pividori <marcos.pividori@gmail.com>
 *   - Keon Kim <kwk236@gmail.com>
 *   - Nilay Jain <nilayjain13@gmail.com>
 *   - Peter Lehner <peter.lehner@dlr.de>
 *   - Anuraj Kanodia <akanuraj200@gmail.com>
 *   - Ivan Georgiev <ivan@jonan.info>
 */

// First, include all of the prerequisites.
#include <mlpack/prereqs.hpp>

// Now the core mlpack classes.
#include <mlpack/core/util/arma_traits.hpp>
#include <mlpack/core/util/log.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/deprecated.hpp>
#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>
#include <mlpack/core/data/normalize_labels.hpp>
#include <mlpack/core/math/clamp.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/math/random_basis.hpp>
#include <mlpack/core/math/lin_alg.hpp>
#include <mlpack/core/math/range.hpp>
#include <mlpack/core/math/round.hpp>
#include <mlpack/core/dists/discrete_distribution.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>
#include <mlpack/core/dists/laplace_distribution.hpp>
#include <mlpack/core/dists/gamma_distribution.hpp>
//mlpack::backtrace only for linux
#ifdef HAS_BFD_DL
  #include <mlpack/core/util/backtrace.hpp>
#endif

// Include kernel traits.
#include <mlpack/core/kernels/kernel_traits.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/polynomial_kernel.hpp>
#include <mlpack/core/kernels/cosine_distance.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/epanechnikov_kernel.hpp>
#include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
#include <mlpack/core/kernels/laplacian_kernel.hpp>
#include <mlpack/core/kernels/pspectrum_string_kernel.hpp>
#include <mlpack/core/kernels/spherical_kernel.hpp>
#include <mlpack/core/kernels/triangular_kernel.hpp>

// Use OpenMP if compiled with -DHAS_OPENMP.
#ifdef HAS_OPENMP
  #include <omp.h>
#endif

// Use Armadillo's C++ version detection.
#ifdef ARMA_USE_CXX11
  #define MLPACK_USE_CX11
#endif

#endif
