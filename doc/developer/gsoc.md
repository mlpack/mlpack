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
   [policy-based design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design#Policy-based_design)
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

When it comes time to write your proposal, see the
[proposal guide](#writing-your-proposal) below.

## Writing your proposal

This section has some details on how you should prepare your proposal.  It is
only a guide, so please do not feel constrained to follow it to the letter!

### Things to think about before applying

Before you consider applying to work on mlpack for Google Summer of Code, please
consider the following points:

Students who are accepted into the Google Summer of Code program are expected to
work full-time hours for the 12 weeks of the program---that's 40 hours a week.
It is not a good idea to try to do GSoC and also work another job; there simply
isn't enough time.  It's okay if your schedule doesn't allow 40 hour weeks every
week (for instance, if you are going on vacation) but you'll have to let us know
so that we can work around it.

mlpack is a complex library implementing difficult machine learning algorithms
and sometimes using confusing C++ language features.  Any GSoC project for
mlpack will almost certainly involve becoming closely familiar with the internal
workings of some of these algorithms.  Although mentors exist to help out with
this process, it's unreasonable to expect them to be able to explain every part
of every paper you might need to read.  So some level of self-motivation and
definitely a willingness to learn are prerequisites.

Participation in the community (via chat, GitHub, or the mailing list; see the
[community page](community.md)) is highly encouraged and helps us get a better
picture of who you are, your abilities, and how you will be able to contribute
to the library over the summer.  In addition, the community is there to help you
out!  Feel free to ask questions, but do keep in mind that mlpack is a fairly
small organization so there may not always be someone around to help.

### Basic information

Once you've considered those three points, and want to write an application,
make sure that your application has the following basic information.  If you've
already made yourself known to the community, make sure any aliases you've used
there are posted here, so that we can know this application corresponds to
someone we have already met.

 * Name
 * University / school
 * Field of study
 * Date study was started
 * Expected graduation date
 * Homepage
 * Email
 * GitHub username, and Matrix/Slack handle (if applicable)
 * Interests and hobbies

Also, be sure to answer the following questions about your technical proficiency
and coding skills:

 * What languages do you know?  Rate your experience level (1-5: rookie-guru)
   for each.
 * How long have you been coding in those languages?
 * Are you a contributor to other open-source projects?
 * Do you have a link to any of your work (i.e. GitHub profile)?
 * What areas of machine learning are you familiar with?
 * Have you taken any coursework relevant to machine learning?

### Optional questions

Here are some questions intended for us to get an idea of who you are.  You
aren't required to answer them, but it is helpful for us.  There is no right
answer; these are open-ended questions.

 * What are your long-term plans, if you have figured those out yet?  Where do
   you hope to see yourself in 10 years?
 * Describe the most interesting application of machine learning you can think
   of, and then describe how you might implement it.
 * Both algorithm implementation and API design are important parts of mlpack.
   Which is more difficult?  Which is more important?  Why?

### The project proposal

The most important part of a proposal is a project proposal.  After including
the information above, you should write a project proposal, which should clearly
outline the following things:

 * The objectives of the proposed project.  This should detail what you expect
   to have accomplished at the end of the summer.
 * Some background information on the project.  This may include descriptions of
   the algorithms / data structures / ideas you plan to implement.  Especially
   for more complex ideas, be sure to detail background information so that a
   person who is reasonably familiar with machine learning and mlpack will be
   able to understand your description without needing to consult other
   references (but, when relevant, do link to other references).
 * If applicable, describe the API of your finished project and how it will fit
   in with the rest of the mlpack code.  We realize that it's impossible to know
   exactly how the API will work when the project is not yet started, but you
   should at least have a basic idea of how you might want it to work.
 * Describe how you will test your project.  Keep in mind that with machine
   learning algorithms, testing is often the most difficult part of the process,
   and it can be very hard to write good tests---especially for statistical
   algorithms.
 * Describe a timeline of some sort for your project, broken down week by week
   (or thereabouts).  In reality we understand that things don't often go
   according to plan, but having a predefined plan can significantly help the
   development process.

### Other information and tips for success

Please feel free to include any other information that you think is relevant but
none of the above questions cover.

Being accepted into Google Summer of Code is a competitive process and the
proposal process may be time-consuming.  Prospective students who have already
contributed code to the library and are participating in the community are more
likely to be selected, as seeing contributions is helpful for displaying your
talent and ability.  The [community page](community.md) is a good place to start
getting involved.

We look forward to hearing from you!
