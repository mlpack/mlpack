---
title: 'mlpack 3: a fast, flexible machine learning library'
tags:
- machine learning
- deep learning
- c++
- optimization
- template metaprogramming

authors:
- name: Ryan R. Curtin
  orcid: 0000-0002-9903-8214
  affiliation: 1

- name: Marcus Edel
  orcid: 0000-0001-5445-7303
  affiliation: 2

- name: Mikhail Lozhnikov
  orcid: 0000-0002-8727-0091
  affiliation: 3

- name: Yannis Mentekidis
  orcid: 0000-0003-3860-9885
  affiliation: 5

- name: Sumedh Ghaisas
  orcid: 0000-0003-3753-9029
  affiliation: 5

- name: Shangtong Zhang
  orcid: 0000-0003-4255-1364
  affiliation: 4

affiliations:
- name: Center for Advanced Machine Learning, Symantec Corporation
  index: 1
- name: Institute of Computer Science, Free University of Berlin
  index: 2
- name: Moscow State University, Faculty of Mechanics and Mathematics
  index: 3
- name: University of Alberta
  index: 4
- name: None
  index: 5

date: 5 April 2018
bibliography: paper.bib
---

# Summary

In the past several years, the field of machine learning has seen an explosion
of interest and excitement, with hundreds or thousands of algorithms developed
for different tasks every year.  But a primary problem faced by the field is the
ability to scale to larger and larger data---since it is known that training on
larger datasets typically produces better results [@halevy2009unreasonable].
Therefore, the development of new algorithms for the continued growth of the
field depends largely on the existence of good tooling and libraries that enable
researchers and practitioners to quickly prototype and develop solutions
[@sonnenburg2007need].  Simultaneously, useful libraries must also be efficient
and well-implemented.  This has motivated our development of mlpack.

mlpack is a flexible and fast machine learning library written in C++ that has
bindings that allow use from the command-line and from Python, with support for
other languages in active development.  mlpack has been developed actively for
over 10 years [@mlpack2011, @mlpack2013], with over 100 contributors from
around the world, and is a frequent mentoring organization in the Google Summer
of Code program (\url{https://summerofcode.withgoogle.com}).  If used in C++,
the library allows flexibility with no speed penalty through policy-based design
and template metaprogramming [@alexandrescu2001modern]; but bindings are
available to other languages, which allow easy use of the fast mlpack codebase.

For fast linear algebra, mlpack is built on the Armadillo C++ matrix library
[@sanderson2016armadillo], which in turn can use an optimized BLAS
implementation such as OpenBLAS [@xianyi2018openblas] or even NVBLAS
[@nvblas] which would allow mlpack algorithms to be run on the GPU.  In
order to provide fast code, template metaprogramming is used throughout the
library to reduce runtime overhead by performing any possible computations and
optimizations at compile time.  An automatic benchmarking system is developed
and used to test the efficiency of mlpack's algorithms [@edel2014automatic].

mlpack contains a number of standard machine learning algorithms, such as
logistic regression, random forests, and k-means clustering, and also contains
cutting-edge techniques such as a compile-time optimized deep learning and
reinforcement learning framework, dual-tree algorithms for nearest neighbor
search and other tasks [@curtin2013tree], a generic optimization framework with
numerous optimizers [@curtin2017generic], a generic hyper-parameter tuner, and
other recently published machine learning algorithms.

For a more comprehensive introduction to mlpack, see the website at
\url{http://www.mlpack.org/} or a recent paper detailing the design and
structure of mlpack [@curtin2017designing].

# References
