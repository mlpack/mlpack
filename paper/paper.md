---
title: 'mlpack 4: a fast, header-only C++ machine learning library'
tags:
  - machine learning
  - deep learning
  - c++
  - header-only
  - efficient

authors:
 - name: Ryan R. Curtin
   orcid: 0000-0002-9903-8214
   affiliation: 1
 - name: Marcus Edel
   orcid: 0000-0001-5445-7303
   affiliation: 2
 - name: Omar Shrit
   orcid: 0000-0002-8621-3052
   affiliation: 1
 - name: Shubham Agrawal
   orcid: 0000-0001-8713-4682
   affiliation: 1
 - name: Suryoday Basak
   orcid: 0000-0002-1982-1787
   affiliation: 3
 - name: James J. Balamuta
   orcid: 0000-0003-2826-8458
   affiliation: 4
 - name: Ryan Birmingham
   orcid: 0000-0002-7943-6346
   affiliation: 5
 - name: Kartik Dutt
   orcid: 0000-0003-3877-0142
   affiliation: 6
 - name: Dirk Eddelbuettel
   orcid: 0000-0001-6419-907X
   affiliation: 7
 - name: Rishabh Garg
   orcid: 0000-0003-0398-0887
   affiliation: 8
 - name: Shikhar Jaiswal
   orcid: 0000-0002-3683-3931
   affiliation: 9
 - name: Aakash Kaushik
   orcid: 0000-0003-1079-8338
   affiliation: 1
 - name: Sangyeon Kim
   orcid: 0000-0003-0717-0240
   affiliation: 10
 - name: Anjishnu Mukherjee
   orcid: 0000-0003-4012-8466
   affiliation: 11
 - name: Nanubala Gnana Sai
   orcid: 0000-0003-0774-7994
   affiliation: 1
 - name: Nippun Sharma
   orcid: 0000-0003-0365-2613
   affiliation: 8
 - name: Yashwant Singh Parihar
   orcid: 0000-0003-3492-0377
   affiliation: 12
 - name: Roshan Swain
   orcid: 0000-0002-7262-8230
   affiliation: 1
 - name: Conrad Sanderson
   orcid: 0000-0002-0049-4501
   affiliation: 13

affiliations:
 - name: Independent Researcher
   index: 1
 - name: Collabora Ltd
   index: 2
 - name: Pennsylvania State University
   index: 3
 - name: Departments of Statistics and Informatics, University of Illinois, Urbana-Champaign
   index: 4
 - name: Emory University
   index: 5
 - name: Delhi Technological University
   index: 6
 - name: Department of Statistics, University of Illinois, Urbana-Champaign
   index: 7
 - name: Indian Institute of Technology Mandi
   index: 8
 - name: Microsoft Research India
   index: 9
 - name: NAVER WEBTOON AI
   index: 10
 - name: George Mason University
   index: 11
 - name: Department of Computer Science and Engineering, IIT Bombay
   index: 12
 - name: Data61/CSIRO and Griffith University
   index: 13

date: 4 November 2022
bibliography: paper.bib

---

# Summary

For over 15 years, the mlpack machine learning library has served as a
``swiss army knife'' for C++-based machine learning [@curtin2013mlpack].
Its efficient implementations of common and cutting-edge machine learning
algorithms have been used in a wide variety of scientific and industrial applications.
This paper overviews mlpack 4, a significant upgrade over its predecessor [@curtin2018mlpack].
The library has been significantly refactored and redesigned to facilitate
an easier prototyping-to-deployment pipeline, including bindings to other languages
(Python, Julia, R, Go, and the command line)
that allow prototyping to be seamlessly performed in environments other than C++.

# Statement of Need

The use of machine learning has become ubiquitous in almost every scientific
discipline and countless commercial applications [@jordan2015machine; @carleo2019machine].
There is one important commonality to virtually all of these applications:
machine learning is often computationally intensive, due to the
large number of parameters and large amounts of training data.
This was the main motivator for the original development of mlpack in the C++
language, which allows for efficient close-to-the-metal implementations [@curtin2013mlpack].

But speed is not everything: development and deployment of applications that use
machine learning can also be significantly hampered if the overall process is
too difficult or unwieldy [@paleyes2020challenges; @lavin2022technology].
Furthermore, deployment environments often have computational or engineering
constraints that make a full-stack Python solution infeasible [@fischer2020ai].
As such, it is important that lightweight and easy-to-deploy machine learning
solutions are available. This has motivated our refactoring and redesign of
mlpack 4: we pair efficient implementations with easy and lightweight
deployment, making mlpack suitable for a wide range of deployment environments.
A more complete set of motivations can be found in the mlpack vision document
[@mlpack2021vision].

mlpack is a general-purpose machine learning library, targeting both academic
and commercial use; for instance, data scientists who need efficiency and
ease of deployment, or, e.g., by researchers who need flexibility and
extensibility.  While there are other machine learning libraries intended to be
used from C++, many, such as FAISS [@johnson2019billion] and FLANN
[@muja2009fast], are limited to a few specific algorithms, instead of a full
range of machine learning algorithms, like mlpack provides.  dlib-ml [@dlib09],
on the other hand, does provide a broad toolkit of machine learning algorithms,
but its extensibility is somewhat limited as it does not use policy-based design
[@alexandrescu2001modern] to provide arbitrary user-defined behavior, and the
range of machine learning algorithms provided is smaller than mlpack's.

# Functionality

The library contains a wide variety of machine learning algorithms,
some of which are new to mlpack 4.  The list of algorithms includes linear regression,
logistic regression, random forests, furthest-neighbor search [@curtin2016fast],
accelerated k-means variants [@curtin2017dual], kernel density estimation [@lee2008fast],
and fast max-kernel search [@curtin2014dual].  There is also a module for
deep neural networks, which has implementations of numerous layer types,
activation functions, and reinforcement learning applications.
Details of the available functionality are provided in the online
[mlpack documentation](https://www.mlpack.org/docs.html). The efficiency of these
implementations has been shown in various works [@curtin2013mlpack; @fang2016m3]
using mlpack's benchmarking system [@edel2014automatic].

The algorithms are available via automatically-generated bindings to Python,
R, Go, Julia, and the command line. Each of these bindings has a unified interface
across the languages; for example, a model trained in Python can be used from
Julia or C++ (or any other language with mlpack bindings).  The bindings are
available in each language's package manager, as well as system-level package
managers such as `apt` and `dnf`.  Furthermore, ready-to-use Docker containers
with the environment fully configured are available on DockerHub, and an interactive
C++ notebook interface via the [xeus-cling](https://github.com/QuantStack/xeus-cling)
project is available on BinderHub.

Once a user has developed a machine learning workflow in the language of their
choice, deployment is straightforward.  The mlpack library is now header-only,
and directly depends only on three libraries: Armadillo [@sanderson2016armadillo],
ensmallen [@curtin2021ensmallen], and [cereal](https://github.com/USCILab/cereal).
When using C++, the only linking requirement is to an efficient implementation
of BLAS and LAPACK (required via Armadillo).  This significantly eases deployment;
a standalone C++ application with only a BLAS/LAPACK dependency is easily
deployable to many environments, including standard Linux-based Docker containers,
Windows environments, and resource-constrained embedded environments.
To this end, mlpack's build system now also contains a number of tools for
cross-compilation support, including the ability to easily statically link
compiled programs (important for some deployment environments).

# Major Changes

Below we detail a few of the major changes present in mlpack 4.  For a complete
and exhaustive list (including numerous bug fixes and new techniques), the
`HISTORY.md` file (distributed with mlpack) can be consulted.

*Removed dependencies.*
In accordance with the vision document [@mlpack2021vision], the majority of the
refactoring and redesign work focused on reducing dependencies and compilation overhead.
This has motivated the replacement of the [Boost](https://www.boost.org)
C++ libraries, upon which mlpack previously depended,
with lightweight alternatives including [cereal](https://github.com/USCILab/cereal)
for serialization.  The entire neural network module was refactored to avoid
the use of Boost (amounting to an almost complete rewrite).
This effort was rewarded handsomely: with mlpack 3, a simple program would often
require several gigabytes of memory just for compilation.
After refactoring and removing dependencies, compilation generally requires
just a few hundred megabytes of memory, and is often an order of magnitude faster.

*Interactive notebook environments.*
mlpack can be used in a Jupyter notebook environment [@kluyver2016jupyter]
via the [xeus-cling](https://github.com/QuantStack/xeus-cling) project.
This is demonstrated interactively on the [mlpack homepage](https://www.mlpack.org).
Examples of C++ notebooks can be found in the
[mlpack examples repository](https://github.com/mlpack/examples),
and these can easily be run on BinderHub.

*New bindings and enhanced availability.*
Support for the Julia [@bezanson2017julia], Go [@pike2012go], and R languages
[@rcore2022; @parihar2022rmlpack] has been added via mlpack's automatic binding
system.  These bindings can be used by installing mlpack from the language's
package manager (`Pkg.jl`, `go get`, `install.packages('mlpack')`).
Furthermore, since mlpack's reduced dependency footprint has significantly
simplified the deployment process, mlpack's Python dependencies are now
available for numerous architectures both on PyPI and in `conda-forge`.

*Cross-compilation support and build system improvements.*
mlpack's build configuration now supports easy cross-compilation, for instance
via toolchains such as [buildroot](https://buildroot.org).  By specifying a few
flags, a user may produce a working mlpack setup for a variety of embedded systems.
This required the implementation of a dependency auto-downloader,
which is capable of downloading [OpenBLAS](https://github.com/xianyi/OpenBLAS)
and compiling (if necessary) for the target architecture.  The auto-downloader
can also be enabled and used for any situation, thus easing installation and
deployment.

# Acknowledgements

Development of mlpack is community-led. It is the product of hard work by
over 220 individuals (at the time of writing). We are also indebted to people
that have provided bug reports over the years.  The development has been supported
by Google, via a decade-long participation the Google Summer of Code program,
and also by NumFOCUS, which fiscally sponsors mlpack.

# References
