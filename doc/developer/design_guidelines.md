Introduction
============

C++ is a very complicated language (especially C++11 and C++14), and we have many free choices when designing a library.  Therefore, it is a good idea to write up a document that details the basic design decisions for mlpack, so that we as developers may use this as a guide to keep the library's code and functionality coherent.  This is also a good place to start for contributors who want to develop large patches to be added to mlpack.  Contributors who only have simple changes to implement are probably best served by looking at the style guidelines.

It is worth remembering that this is a guide, not a rulebook: everything written here is open to some amount of interpretation and the guidelines may apply to some situations well and some situations not at all.  When in doubt, start a conversation.  When you disagree with something written here or think it can be improved, start a conversation.  This is meant to be a living document, so nothing is set in stone---just Markdown.

Goals of mlpack (plus a little history)
=======================================

The mlpack project was started around 2007 as "FASTLIB/MLPACK", a machine learning library for use by Alex Gray's FASTLab (<http://www.fast-lab.org/>) to implement their fast machine learning algorithms.  The types of problems considered were generally statistical tasks: nearest neighbor search, density estimation, range search, and so forth.  In addition to being a research delivery vehicle for the research of the FASTLab, the goal was to produce very high-speed implementations of these algorithms that could be used as building blocks for higher-level tasks.

In 2009 and 2010, a complete code overhaul was started after an in-depth survey of the code.  This led to the design document "The Future of MLPACK": <http://www.igglybob.com/mlpack_future.pdf>.  That document, and the later papers (at the NIPS BigLearning workshop (<http://www.ratml.org/pub/pdf/2011mlpack.pdf>) and in JMLR (<http://www.ratml.org/pub/pdf/2013mlpack.pdf>)) formalizes four development goals of mlpack:

* Implement scalable, fast machine learning algorithms 
* Design an intuitive, simple API for users who are not C++ experts
* Implement as large a collection as possible of machine learning methods
* Provide cutting-edge machine learning algorithms that no other library does

This last goal is somewhat in contrast to the scikit-learn project, which generally only implements stable, well-known algorithms.  mlpack can fill a niche by providing high-quality implementations of algorithms that just appeared in conferences or journals.  In those cases where mlpack is implementing well-known algorithms (i.e. SVMs or other standard techniques), we should strive to ensure that our implementation is faster than other implementations.  To ensure that, we may use the automatic benchmarking system; see https://www.github.com/zoq/benchmarks/.

General Machine Learning Abstractions
=====================================

Machine learning algorithms can operate on a wide variety of data, including numerical, textual, and categorical data.  Because our underlying matrix type is an Armadillo matrix, which does not support textual or categorical data, we assume that all of our data is numerical and can be represented by a matrix (specifically, as an Armadillo matrix, whether it be sparse or dense).

Usually, machine learning algorithms perform some predictive task: they may cluster a set of points; they may perform classification; they may estimate a density; they may find a nearest neighbor.  These are only some of the possible tasks that algorithms in mlpack may accomplish.  Other core components in mlpack include things like probability distributions, metrics, and kernels.

For each of these pieces in mlpack, some preprocessing may be required or useful to accelerate the core functionality of the component, be it classification, nearest neighbor searching, density estimation, regression, evaluating a metric, estimating a probability distribution, or whatever else.  One example is that a Gaussian distribution can do some preprocessing to store the inverse of the covariance matrix, which accelerates the later estimation of probability at a given point.

We will aim to perform the preprocessing in the constructor, re-calculate any preprocessing with accessors and mutators, and then perform the actual processing with a specific method (or set of methods).  To elaborate on each of these points:

* The constructor of an mlpack object *must* return a _valid_ object which is ready to be used (and/or trained).  For a machine learning method, any training should not take place in the constructor: instead, constructing the object should set parameters to train with (note that it is often useful to provide a constructor that calls a training function too).

* mlpack objects should provide copy and move constructors and operators (if different behavior than the default versions are needed); these functions should have warnings in their documentation if the computational cost may be memory-intensive (i.e. if there is a big copy).

* mlpack objects should be serializable via a `Serialize()` function (see the section on serialization).

* Any preprocessing should be done in the constructor.  For example, the `GaussianDistribution` class calculates (1 / sigma^2) and stores that during preprocessing.

* Members of the class should be modifiable through accessors and mutators; when necessary, a mutator should re-calculate any preprocessing that needs to be done.  Mutators should not incur re-training of a model (or re-estimation of parameters); the user should do that manually.

* Machine learning algorithms that have a specific training step (for instance: logistic regression) should expose a `Train()` or `Estimate()` function.  Some machine learning algorithms will learn in batch while others will learn using a single point at a time.  Ideally, both interfaces (batch learning and single-instance learning) should be provided, and batch learning should re-train the model from scratch whereas single-instance learning should update the existing model.  The same applies for other components that need to be estimated (i.e. distributions).

* Evaluation of a machine learning algorithm (functions like `Cluster()`, `Classify()`, `Regress()`, and `Predict()`) should have the same interface across all algorithms.  These should work on both a single point and many points at once, but in general this should be accomplishable via a single function that takes an Armadillo matrix; in the single-instance case, that matrix should have just one column.

* Components of mlpack should be modular via templates, allowing the user to specify their own options.  In accordance with this, no constructor of an mlpack component should take parameters for a template member's constructor.  Instead, there should be a constructor which allows the user to pass in their own already-instantiated template member.

C++ Features
============

Templates
---------

C++ templates are the means by which we achieve fast code.  The primary usage is via 'policy-based design', which allows us to write highly customizable code.  This is easiest to demonstrate with an example.  Consider the following code:

```
template<typename MetricType = EuclideanDistance>
double CalculateAverageDistance(const arma::mat& data, MetricType& metric)
{
  // Calculate the average pairwise distance between points.
  double result = 0.0;
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    for (size_t j = 0; j < data.n_cols; ++j)
    {
      if (i == j)
        continue;
      else
        result += metric.Evaluate(data.col(i), data.col(j));
    }
  }
}
```

This snippet allows the user to pass *any* `MetricType` class so long as it has an `Evaluate()` function which takes two Armadillo vector objects.  Therefore, a user can now use `CalculateAverageDistance()` with the default `EuclideanDistance`, any other metric provided by mlpack, or a metric that they have written themselves.

Each time a template parameter is given with the intention of using policy-based design, it must be made clear in the documentation exactly what methods that template type should have and how they must work.  So, for instance, in the example above, the `MetricType` class should specify that an `Evaluate()` function is necessary, and also that `Evaluate()` between two points must return the evaluation of a valid metric (so, it must be nonnegative, symmetric, and satisfy the triangle inequality).

Inheritance
-----------

In general, we should avoid inheritance (or, at least, virtual inheritance) in deference to templates (see above).  This is because virtual functions incur runtime overhead, and especially in critical inner loops where these functions are called many, many times, this overhead is non-negligible.

Compelling cases for inheritance should be considered carefully.  In order to keep the code consistent, we should be careful with the use of various C++ features: if a user is used to mlpack using templates for static polymorphism, then it is not good if that user suddenly encounters inheritance used to solve the same type of design problem in a different context.

Multiple inheritance isn't used inside the mlpack codebase, and given the previous two paragraphs, is probably not necessary in light of the use of templates.

References and pointers
-----------------------

Where possible, references are preferred; const references are for input parameters (when possible), and non-const references are for output parameters.  Only primitive types should ever be passed by value in a method call, so as to avoid unnecessary object copies.

The use of references allows us to skip checks for NULL because we are making the assumption that the user is passing in a valid object.  Below is an example:

```
template<typename MetricType = EuclideanDistance>
double CalculateOffsetAverageDistance(const arma::mat& data, MetricType& metric, const double offset)
{
  // Calculate the average pairwise distance between points.
  double result = 0.0;
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    for (size_t j = 0; j < data.n_cols; ++j)
    {
      if (i == j)
        continue;
      else
        result += metric.Evaluate(data.col(i), data.col(j)) + offset;
    }
  }
}
```

In this case, all three parameters to the method are input, but we cannot label `MetricType` as const, because a call to `MetricType::Evaluate()` may be non-const -- for instance, if the class is counting how many times `Evaluate()` has been called.

One instance where references don't make sense are with structures like trees.  Consider a simplified binary tree, given below:

```
struct SimpleTree
{
  SimpleTree* left;
  SimpleTree* right;
};
```

In this case, it does not make sense to have `left` and `right` be of type `SimpleTree&`, because various operators on the `SimpleTree` structure may modify the left and right children (for instance, rebalancing of a tree may attach a child somewhere else in the tree).  References can't be re-seated, so rebalancing a reference-based tree would be a nightmare.

So, the general rule of thumb is: always try to use references, but when this isn't reasonably possible (i.e. when you may need to reassign), use pointers.  Note that the use of references can make serialization difficult, so it is not always possible to cleanly use references.

Template metaprogramming and SFINAE
-----------------------------------

SFINAE (substitution failure is not an error) is in wide use throughout the mlpack codebase, in order to perform partial specializations.  So, here is an example, using some structures from Boost:

```
// This overload catches the case where T is a class.
template<typename T>
void Print(T& t,
           const typename std::enable_if<std::is_class<T>::value>::type* = 0)
{
  std::cout << "T is a class!" << std::endl;
}

// This overload catches the case where T is not a class (it is a pointer, reference, or primitive type).
template<typename T>
void Print(T& t,
           const typename std::enable_if<!std::is_class<T>::value>::type* = 0)
{
  std::cout << "T is not a class!" << std::endl;
}
```

Because template metaprogramming is confusing and hard to read, it is very important to be explicit in the documentation.  Here, each overload is documented for the cases that it catches.  If you are going to use template metaprogramming to solve a problem in mlpack, first make sure that the problem can't be solved in some easier way, and second make sure that all aspects of your solution are comprehensively documented.

There do exist some useful utilities for template metaprogramming in `src/mlpack/core/util/`.

Exceptions
----------

We can use the same approach to exceptions that was documented in "The Future of mlpack".  The use of exceptions does not generally incur any significant runtime overhead.  In the context of machine learning algorithms, an exception will generally mean malformed data or erroneous code.  In this situation it is reasonable to throw an exception, since in general it's not really possible to recover from this situation.

Exceptions shouldn't be used as a substitute for a success indicator, though; if a method can easily return `true` or `false` to indicate its convergence, there is no need to throw an exception when convergence does not occur.

Class members
-------------

Any mlpack class should hold no more members internally than are absolutely necessary.  So, because each machine learning algorithm that has a training process will have a `Train()` method that takes the training dataset as input, then it is likely that the class does not need to hold a reference to the training dataset internally, because the training dataset is only used at training time.  (There are exceptions where this is not the case.)

There are some cases where members need to be held internally, but copying these members would be costly, and holding a reference internally is not necessarily the best idea.  Consider the following example:

```
template<typename OptimizerType>
class LogisticRegression;
```

This is a logistic regression class, which uses the OptimizerType optimizer to train the model.  But, it is possible that the OptimizerType class is quite large, or holds matrices internally, so we want to avoid copies.  At the same time, we want to hold an instantiated OptimizerType internally, because we will be using it.  The user should also be able to pass in their own OptimizerType to use.  This can be accomplished in the following way:

```
template<typename OptimizerType>
class LogisticRegression
{
 public:
  LogisticRegression(...) : optimizer(new OptimizerType()), ownsOptimizer(true), ... { ... }
  LogisticRegression(..., OptimizerType& opt) : optimizer(&opt), ownsOptimizer(false), ... { ... }
  ~LogisticRegression()
  {
    if (ownsOptimizer)
      delete optimizer;
    ...
  }

  const OptimizerType& Optimizer() const { return *optimizer; }
  OptimizerType& Optimizer() { return *optimizer; }

 private:
  OptimizerType* optimizer;
  bool ownsOptimizer;
};
```

In general, this type of pattern should be applied to prevent copies of constructor parameters, as well as allowing user flexibility: with this design, a user can create a default optimizer and modify its parameters via `Optimizer()`, or create their optimizer on their own and pass it in via the constructor.

Requirements of each class
--------------------------

Each mlpack class should be serializable, via a `Serialize()` function with the signature:

```
template<typename Archive>
void Serialize(Archive& ar, const unsigned int version);
```

The particular usage of this class follows the `boost::serialization` library guidelines, and functions equivalently to the `serialize()` method in regular `boost::serialization` classes.  For coherence with the method naming guidelines, mlpack implements a shim (in `src/mlpack/core/data/`) that allows a `Serialize()` method to be used in place of a `serialize()` method.

In order to serialize properly, most objects will need to avoid the use of references entirely, because serialization generally means that all data members must have their values changed (during loading, at least).  In those situations where references *must* be used, or where a default constructor is not available, it is possible to have `boost::serialization` load into a pointer; see `src/mlpack/core/tree/binary_space_tree/binary_space_tree_impl.hpp`.

This serializability allows easy loading and saving of models.

Style Guidelines
================

Tabbing
-------

Tabs should be two spaces wide.  i.e.

```
void Class::Method(int num)
{
  DoSomething();

  for (int i = 0; i < num; i++)
    DoSomethingElse(i);
}
```

Use spaces, not tab characters.  See [jwz's rant](http://www.jwz.org/doc/tabs-vs-spaces.html) for more information and reasoning.  If you're using vim, here is what should go in your .vimrc:

```
set softtabstop=2
set expandtab
set shiftwidth=2
```

One exception to this rule is the public: and private: keywords in classes, which should be indented just one space: 
	 	 
``` 
class X 
{ 
 public: 
  // Stuff. 
 
 private: 
  // Stuff. 
}; 
``` 

Spacing for operators
---------------------

There should be a space between an operator and its argument.  For instance:

```
if (condition)
```

or in the case of a for loop,

```
for (i = 0; i < 10; ++i)
```
 
Line Length and Wrapping
------------------------

Lines should be no more than 80 characters wide.  This was chosen because it is a standard for other projects out there, and it is not too difficult to adhere to.  Also, it makes putting code samples in a paper possible and easy.

When wrapping a line, the next line should be tabbed twice from where the previous line began.  An example is easier to understand:

```
void Class::Method(arma::mat& xx, const arma::mat& y)
{
  // We assume local variables arma::vec z and m.

  xx = y * SomeReallyLongMethodName(z, m) + 
      SomeOtherReallyLongMethodNameThatIsLong(z, m) +
      AThirdReallyLongMethodNameThatIsAlsoVeryLong(z, m);
}
```

As a side note, try to avoid method names that are that long if possible.

Method Declarations
-------------------

Suppose you have a method with parameters that ends up being longer than 80 characters.  Wrap it such that each new line of parameters lines up with the previous line of parameters, like so:

```
AllNN::AllNN(arma::mat& referencesIn,
             struct datanode* moduleIn,
             bool alias_matrix,
             bool naive)
```

If the method names are so long that you can't make that fit, you can do this:

```
AllNN::AllNN(
    arma::mat& referencesIn,
    struct datanode* moduleIn,
    bool alias_matrix,
    bool naive)
```

And if the method and class names are too long for that, you could do this:

```
void AReallyLongClassName::
SomeReallySuperLongMethodNameWhichShouldBeShorter(
    arguments)
```

If you add templates too, you can do this:

```
void KMeans<
    DistanceMetric,
    InitialPartitionPolicy,
    EmptyClusterPolicy>::
Cluster(const arma::mat& data,
        const size_t clusters,
        arma::Col<size_t>& assignments)
```

Constructor Initialization Lists
--------------------------------

We have already defined how parameters should be, but initialization lists should adhere to the line wrapping style with two tabs.  The same example is below:

```
AllNN::AllNN(arma::mat& referencesIn,
             struct datanode* moduleIn,
             bool aliasMatrix,
             bool naive) :
    module(moduleIn),
    naive(naive),
    references(referencesIn.memptr(), referencesIn.n_rows,
        referencesIn.n_cols, !aliasMatrix),
    queries(references.memptr(), references.n_rows, references.n_cols,
        false)
{
  // Do some stuff inside of the constructor here that could not be done with
  // initialization lists.
}
```

Again, it may be easier to read each initialization list entry on separate lines if many things are being initialized.  That is left up to the developer to decide.

Note that on very long initialization list lines, the wrapping rules are again used.

Brace Placement
---------------

Avoid placing opening braces on the same line as a `loop/if/else/switch/function/class`.  The block is more readable with braces aligned.

Use
```
if (n > 12)
{
  x += 2;
  y -= 12.5;
}
```

rather than
```
if (n > 12) {
  x += 2;
  y -= 12.5;
}
```

To conserve lines, refrain from using braces if the `if`, `while`, or `for` statement applies to a single statement.

Namespaces
----------
 
The exception to the rules of braces and tabbing is namespaces.  For simplicity, write this: 
 	 
``` 
namespace mlpack { 
namespace mvu { 
	 
// stuff goes here 
 	 
} // namespace mvu 
} // namespace mlpack 
```
	 	 
Don't indent anything inside namespaces---it's a waste of columns, especially when you are several namespaces deep. 
 
Operator and Keyword Spacing Rules
----------------------------------

Use spaces between your operators for readability.  Below is an example:

```
for (int i = 0; i < 10; i++)
{
  if (i > 5)
    someValue = (4 * i) + 5;
  else
    someValue = (someBool ? (3 * i) : 2);
}
```

Things like `i=(4+5-i)` are not cool, and are not easy to read.  A space between parentheses and what is inside of them, like `i = ( 4 + 5 - i )` is technically superfluous but is still readable so is not a huge problem.  Also do note the spaces between the for and if operators and the opening parentheses.

Naming Conventions
------------------

Use [camel casing](http://en.wikipedia.org/wiki/CamelCase) for all names.  Capitalize class and method names.  Variable, reference, and pointer names begin with a lower case letter.  Setters and getters share names with their respective data members, distinguished by the case of the first letter of each of their names. 

```
class Hilarious
{
  private:
    int laugh;
    char chuckle;
    matrix& matr;
  public:
    Hilarious();
    const int Laugh() { return laugh; }
    void Laugh(int i) { laugh = i; }
    const matrix& Matr() { return matr; }
}
```

It is your choice whether you want to provide a setter that takes the input as an argument (like `void Laugh(int i)`) or if you want to provide a setter that allows direct access (like `int& Laugh()`).  Generally the latter is used when it is not necessary to validate what the user is setting the value to (or if no processing is necessary on the value), and the former is used when validation or processing is necessary.

Pointers and reference placement
--------------------------------

Place the reference (`&`) or pointer (`*`) with the type, not with the variable:

```
const A& b; // not const A &b
const C* d;
```

Casting
-------

When doing a C-style cast, use spaces:

```
double d = (double) integer;
```

and not

```
double d = (double)integer;
```

Comments
--------

**Everyone has their own individual style for comments.  This section doesn't intend to hinder your own personal style (unless your style is 'no comments').**

Not everyone that uses mlpack is an English speaker, so it's helpful to always have comments that are complete sentences with proper grammar and punctuation; this should help any translation (whether done by human or machine).  So try to avoid ambiguous short phrases that are difficult to translate.

Also, be aware that not everyone reading the code will be an expert on the technique, so be sure to insert high-level comments on what the algorithm is doing (at the same time, saying `Do an eigendecomposition of X` every time you write `X.eig()` is overly verbose).  Remember that sooner or later someone will come along to maintain your code (it might even be you) and have nothing more than a basic machine learning and C++ background.

As always, more comments are better than fewer comments.

Citations
---------

Following a lengthy discussion in [#195](https://github.com/mlpack/mlpack/issues/195), consensus was that there should be

* no citations in the `-h` output of any mlpack program (that is, no citations in `PROGRAM_INFO()` macros).
* citations in comments should be in BiBTeX format (keep it simple -- author, year, pages, title, journal/conference; no need for the DOI or URL or abstract or anything).  One exception for URLs is arXiv papers; for instance, ICLR does not have proceedings; all the papers are on arXiv.  So we can assume that arXiv URLs won't go out of date and therefore they should be ok to put in the code.


Lambda Functions
----------------

Lambda functions can be used to encapsulate code that is passed to another method. The brace placement will follow the same guidelines as other functions i.e. The opening braces of lambda function will also be indented. An example of lambda function is shown below.
```
digits.erase(std::remove_if(digits.begin(), digits.end(), [&blacklist](int i)
    {
      return blacklist.find(i) != blacklist.end();
    }),
  digits.end());
```