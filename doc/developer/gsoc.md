# mlpack and Google Summer of Code

mlpack is a proud participant in Google Summer of Code. We have been a part of
the program since 2013, with over 50 students accepted over the years for mlpack
(all of whom succeeded in the program).  Each year, we receive very many
applications and it is a competitive process, so this page exists to help you
determine if you could be a strong candidate.

## How to be a good candidate

The two most important qualities that an mlpack GSoC candidate can possess is
the ability to be self-sufficient and the willingness to learn.

Self-sufficiency is key because mentors have limited time and can’t put in as
much time helping the student as the student is putting in. However, this of
course does not mean that a student (or a prospective student) can never ask any
questions! mlpack is a complex library and can sometimes take help and
explanation to understand.

A willingness to learn is also important because virtually every potential
project will require the student to become familiar with a new algorithm or C++
technique. mlpack is a complex library with many components and it is likely
during your project that you might have to use or interact with some other part
of the codebase.

## Things to be familiar with

Of course, these are not the only important ingredients for a successful GSoC
project. A student should ideally be familiar with

 - *open source software development*: opening pull requests, using git, opening
   issues. mlpack uses Github, which has great documentation. You can learn
   about the workflow using [this link](https://docs.github.com/en), if you are
   not already familiar.

 - *using the development toolchain on your computer*: you should be able to
   download and compile mlpack, make changes to the code, and recompile with the
   new changes. There is a [guide](../user/install.md) for how to build mlpack
   and that would be a great place to get started. If you’re on Windows, then
   the [Windows build guide](../user/build_windows.md) could be very useful. See
   also the [Community page](community.md) for more information on getting
   started.

 - *at least intermediate C++ knowledge*: mlpack uses lots of different C++
   paradigms including template metaprogramming, C++ features like rvalue
   references, and other strategies, in order to make the code fast. You should
   be at least familiar with some of these language features and what templates
   are, even if you have not used them in-depth, so that you can understand the
   mlpack codebase. Some examples of patterns that are often used inside of
   mlpack are SFINAE ([example in mlpack, see std::enable_if usages](https://github.com/mlpack/mlpack/blob/565cfd3aad22deec0656b86e801052593a937723/src/mlpack/methods/mean_shift/mean_shift.hpp)),
   [policy-based design](https://www.drdobbs.com/policy-based-design-in-the-real-world/184401861),
   and [compile-time class traits](https://accu.org/xaraya/journals/442.html).
   Here are some [other useful resources](https://en.wikipedia.org/wiki/Template_metaprogramming)
   for learning template metaprogramming, and some useful
   [reference books](https://www.aristeia.com/books.html).
   If some of this sounds new to you, don’t feel overwhelmed; it’s not a
   necessity, but it is helpful. You should at least be prepared to learn about
   it!

 - *project-specific knowledge*: you’ll need to have an in-depth understanding
   of the specific project that you choose. If you’re not sure what project you
   want to work on, see the
   [SummerOfCodeIdeas](https://github.com/mlpack/mlpack/wiki/SummerOfCodeIdeas)
   wiki page, which has ideas for GSoC projects that you might find interesting.
   But you aren’t required to do one of those projects; if you have another
   interesting idea, propose it and see if a mentor is interested in supervising
   the project! If you have any questions about a project, be aware that it’s
   possible that the question has already been answered on Github. Take a look
   through the closed issues or search to see if there’s already an answer to
   your question.  Get involved!

For a strong proposal, it’s important to be a part of the mlpack community, via
contributions, code reviews, helping others solve their issues, and so forth.
There are some pointers on the [community page](community.md) on how to start
contributing and get involved.

When it comes time to write your proposal, we do have a
[proposal guide](https://github.com/mlpack/mlpack/wiki/Google-Summer-of-Code-Application-Guide)
that you can take a look at to guide your application.
