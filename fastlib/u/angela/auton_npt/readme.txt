PLEASE DO NOT DISTRIBUTE IN ANY WAY RIGHT NOW. CODE NOT FULLY TESTED
AND NOT NECESSARILY FULLY OPTIMIZED.


1. Auton lab programming....

(1) in your CVS "h" directory do

      ./localize-dirlib npt
      cd npt
      gmake localize
      gmake t=fast

  To run npt do:

    ./npt npt in <datafile> [options]

2. Other people programming...

You can tell your user to do the following:
 (1) Copy npt-notyetpublic.tar.gz to a directory under which they want
     the files to be created.
 (2) gunzip npt-notyetpublic.tar
 (3) tar xvf npt-notyetpublic.tar
       (This will create the single directory npt-notyetpublic)
 (4) cd npt-notyetpublic/npt
 (5) gmake 
        (That will create a debug version of the whole program. The 
         executable will be npt-notyetpublic/npt/npt, which will in 
         fact be a symbolic link to an executable in a debug subdirectory)
 (6) Instead (or in addition), for speed, type
     gmake t=fast
 (7) For general information on the way the software is set up, and some of 
     our common datatypes (especially dyv's (dynamic vectors), dym's
     (dynamic matrices), ivec's (dynamic vectors of integers), string_array's,
     and the Auton's Graphics (draw/amgr.h) and command-line-based
     applications (utils/command.h) see:
       http://www.cs.cmu.edu/~AUTON/programming.html

3. Command line options

Use:

./npt npt in <datafilename> [options]

where datafilename is the name of an ascii file in which 
each datapoint is given on a separated line, and each line
is a series of numbers specified by spaces or commas. 

Optionally the datafile may have the first row be a set of
non-numeric strings correspnding to attribute names.

So, for example, .csv (comma separated values) files are fine.

More file format details in ../dset/{ds,dsut}.h

Options are....

ARGV              num_rows    <int> default 999999999 
     If num_rows is less than #points in dataset, will use a
     random sample of dataset containing "num_rows" points. Useful
     if you're doing experiments vs dataset size.
    
ARGV      all_equal_metric   <bool> default     TRUE 
     Do we use the plain obvious distance metric? (TRUE means
     "don't autoscale the dimensions") (FALSE means "put everything
     inside a unit cube")

ARGV                matcher <matchspec>  default     0.05 
     Specify the n-point predicate to use. The syntax is described
     below.

ARGV        thresh_ntuples <double> default        0
     Skip any n-tuples of kdnodes in which the maximum possible
     number of tuples from them are less than thresh_ntuples. 
     This argument is IGNORED if autofind is set to TRUE

ARGV        connolly_thresh <double> default        0
     A simple very fast method of pruning suggested by Andy
     Connolly: Skip any n-tuples of kdnodes in which the maximum
     diameter of the knodes <= connolly_thresh

     With a value of 0 (the default) this parameter has no effect

     You can still be assured that the true count lies between
     whatever bounds we return

     This argument is IGNORED if autofind is set to TRUE

ARGV                     n    <int> default        2 
     The "n" in "n-point"

ARGV              autofind   <bool> default    FALSE 
     If set to TRUE do repeated attempts at solving with successively
     smaller thresh_ntuples values, until you find one in which the
     maximum possible error is within fraction "errfrac" of the true
     count.

ARGV               errfrac <double> default     0.05 
     Only relevant if autofind is TRUE. This is "epsilon" in the paper.

ARGV               verbose <double> default        1 
     Set to 0 unless you want an animation (which'll slow it down)

ARGV                  rmin    <int> default       20 
     The leaf-list size in the kd-tree. Unclear what its general effect is,
     but 'smaller the better subject to fitting in main memory' is NOT
     the way to go. I suggest not playing with this. 20 usually is fine.

ARGV         min_rel_width <double> default   0.0001 
     kdnodes smaller than this are not split no matter how many points.
     I suggest not playing with this.

ARGV        rdraw <bool> default FALSE
     If set to true, pops up a window and does a very fast flickery
     animation of progress in the search

ARGV               winsize    <int> default      512 
       Pixels in ag window edge (size of popup window)

ARGV               binfile    <filename>   default NULL

    (The following courtesy of Nick Konidaris)
    A binfile is a /single line/ that looks like this:
    <field1> <field2> <field3> ... <fieldN>

    Where 
        <fieldi> is a parameter that you would pass to the switch matcher.  An
        example binfile looks like:

        1,2 2,3 3,4 5,6 6,10 10,100

    And it will iteratively go through each of these fields to find matches.

    AWM Notes: If people start using binfiles a lot, it would be worth making
               some pretty simple algorithmic improvements (described in
               the Gray and Moore paper) to make this go much faster
               by searching all bins at once instead of one at a time.

ARGV            rdata <string> default random.csv
     If you want to do an n-point count between two datasets (e.g. data
     and random) then this is the argument with which you can specify the
     random dataset.

ARGV             use_permute <bool> default TRUE
     You should not need to use this.  By setting it to FALSE you can get
     the same behavior that used to appear in the "compound" predicate cases.
     i.e. a set of points is counted multiple times if it matches the 
     template in multiple orderings and individual points can appear in the 
     same tuple multiple times.
     In its default (TRUE) setting, it now counts each set of points only 
     once and a set must consist of unique points.

ARGV                format <string> default   (n d's, i.e. "ddd...d" )
     If you are doing 2-point, use format dd for data vs data
                                   format dr for data vs random
                                   format rr for random vs random

     If you are doing 3-point, use (for example) format ddr 
	 for data vs data vs random etc...

     Note: the d's must be at the beginning and the r's must be at the end
     of the string.

ARGV                sametree <bool> default FALSE
     When TRUE, it will construct a kd-tree based on the union of the data
     and random data points (if both are present).  Then it stores the data
     and random points each in their own tree, but keeps the splits and
     bounding boxes from the kd-tree created with the union.  The result is
     that the bounding boxes may be loose and thus the pruning will be 
     inefficient, but it guarantees that the trees carve up the space in
     exactly the same way for data and random points.  Just leave this as
     its default, FALSE, unless you're really digging into the internals of
     the tree structures.

ARGV                nweights <int> default 0
     The last nweights columns in the data set are assumed to be weightings
     on the data.  If this argument is non-zero, all npoint operations will 
     return both the counts of matching n-tuples, and the weighted counts.
     Each tuple matching the template will be counted with a weight equal to 
     the product of the weights of the points in the tuple.
     Note: the approximation methods still use the plain counts to guide the
           coarseness of the approximation.
     Caveat: The weighted npoint computation currently only works for n <= 3.
             It can be extended for higher n if necessary.
             Implementation note for future reference:  Some possibilities
             for improving this (ranging from extending the current hack to
             modifying the algorithm) are:
             1. more "corrected counting computation" and caching for higher n.
             2. force recursion until there are no sets of identical leaves
                greater than size 3.
             3. Eliminate current use of upper bounds at every step.

     *** HORRIBLE WARNING ***

         Okay there's something ugly going on. For efficiency, if the
         the code ever sees that the same dataset is used in all components
         of the format, it uses symmetry to avoid wasting time doing 
         redundant counting (e.g. in 2-point it doesn't count the same
         pair twice).

         Of course, there's no such kind of symmetry with, say, DR, type
         searching.

         The warning is: MAKE SURE YOU TAKE THIS INTO ACCOUNT IF YOU 
         EVER TRY TO DO COMPARISONS OF A DD vs DR vs RR COUNT.

ARGV                do_wsums <bool> default FALSE
ARGV                do_wsumsqs <bool> default FALSE

         Setting this to TRUE will yield an additional computation that is
         identical to the standard weights computation except that each tuple
         matching the template will get a value that is the sum (of squares
         for do_wsumsqs) of the weights of each data point in the tuple 
         (rather than the product as with the standard computation).  The 
         resulting sums will be printed out in additional to the other weight 
         outputs


-----------------------------------------------------------------------------

Doing the kdtree animation demo:

cd ......../h/mrkd

     gmake t=fast
     ./mrkd drawsearch in <filename> [options]

ARGV              num_rows    <int> default 999999999 
      Same as with "npt"

ARGV            searchtype <string> default    count 
       Available searchtypes are:
           nn search count 
       This allows you to choose between demoing reange search, range count,
       and nearest neighbor

ARGV               winsize    <int> default      512 
       Pixels in ag window edge

ARGV                  rmin    <int> default        5 
       As above

ARGV         min_rel_width <double> default   0.0001 
       As above

then follow instructions printed to stdout

4. The "Multiple 2 Point" (m2p) option

   This efficiently creates the data for a histogram of counts
   for 2-point correlation. Both the counts and the separations
   are loaded and represented in the log domain

Syntax:  ./npt m2p in <filename> <stdoptions> <m2poptions>

  The standard options (described in 3. above) are...

   num_waitforkey_skipsn <number>
   in <filename> - the datafile
   rmin <n> - kdtree leaf num points
   min_rel_width <sep> - kdtree leaf min radius
   num_rows <n> - Allows random subsample of datapoints
   rdata <datafilename> - The "Random" data for RD computations
   format <string> - Something like "rd" if using random data

   winsize <pixels> (for the graphical animation)
   ticks

   Ticks are the most important notion. They are values on the x-axis
   of the histogram indicating where one bar begins and the next ends.
   The number of bars (actually called number of buckets in the code)
   is one less than the number of ticks, because there is one tick
   marking the leftmost side of the lowest bar and another marking the
   right side of the highest.

   Ticks can be specified in two ways

      (1) tickfile <filename>
           Where <filename> is a text file containing whitespace 
             separated numbers (or numbers on different lines in the file)
             where the i'th number corresponds to the i'th tick.
      (2) implicitly

         low_log_sep <number>
         high_log_sep <number>
         num_buckets <number>

           These specify the leftside of the lowest bucket, the right
           side of the highest bucket, and the number of buckets.

   errfrac <numberbetwen 0 and 1>
       Tells the computer it need not get the exact height of the bars
       accurate above a given percent of the height of the highest bar.
       IMPORTANT: This can mean the different between 20 mins and 20 secs
                  runtime. Use large errfrac initially to get a feel for
                  the shape of the curve.

   outfile <filename>
       Optionally dump the histogram bar heights to this file.

-----------------------------------------------------------------------------

Matcher specification syntax (see matcher.h for up-to-date info)

/* A matcher data structure is used to describe exactly what kind of
   n-point correlation predicate we're using. 

    Predicate parameters depend on which kind of n-point predicate we are
    using. Here are the four available types:

       1. scalar-threshold
       2. scalar-between
       3. compound-threshold
       4. compound-between

    Now let's go through the 4 types in more detail.
   
       1. scalar-threshold

        Represented on the command line as: matcher <number>
        Example:                            matcher 0.2

        This matcher matches an n-tuple of points if and only if all
        pairs of points (x_i,x_j) in the tuple satisfy

             Dist(x_i,x_j) <= <number>

 
       2. scalar-between

        Represented on the command line as: matcher <number>,<number>
        Example:                            matcher 0.2,0.6

         WARNING: Either there must be no space between the , and the numbers,
                  or the expression should be in quotes:
 
                     This is fine:  matcher 0.2,0.6
                     This is fine:  matcher "0.2 , 0.6"
                     This is bad:   matcher 0.2 , 0.6
                     

        matcher p,q matches an n-tuple of points if and only if all
        pairs of points (x_i,x_j) in the tuple satisfy

             p <= Dist(x_i,x_j) <= q

       3. compound-threshold

           Represented on the command line as: matcher <filename>
           Example:                            matcher 3p.predicate
           Where 3p.predicate is an ascii file containing a matrix, e.g:

               0    0.1   0.5
               0.1  0     0.2
               0.5  0.2     0

           An n-tuple (x_1,x_2, .. x_n) matches the compound-threshold
           matrix H if and only if

             forall i in 1..n, and all j in i+1...n, Dist(x_i,x_j) <= H[i][j]

           The example above would match triangles in which the first two
           points were within distance 0.1 of each other, the first and third
           within distance 0.5 and the second and third within 0.2.

           Error Checking:
 
              The file MUST contain a matrix with n lines, and n numbers
              (space or comma separated) on each line, with the j'th element
              on the i'th line representing H[i][j]

              H must be symmetric

              H must have a zero diagonal

              All other entries must be strictly greater than zero.

                  
       4. compound-between

           Represented on the command line as: matcher <filename>,<filename>
           Example:                            matcher 3lo.txt,3hi.txt

           Where 3lo.txt and 3hi.txt could be, for example...

           3lo.txt:

               0    0.1   0.5
               0.1  0     0.2
               0.5  0.2     0

           3hi.txt:

               0    0.2   0.9
               0.2  0     0.21
               0.9  0.21     0

           An n-tuple (x_1,x_2, .. x_n) matches the compound-threshold
           matrix pair L,H if and only if

             forall i in 1..n, and all j in i+1...n, 
                 L[i][j] <= Dist(x_i,x_j) <= H[i][j]


   SYMMETRY: This SYMMETRY statement is superceded by the next one.
             There is an important difference in the way that "scalar" versus
             "compound" predicates are counted.

             A scalar predicate neglects redundant permutations of points,
             thus if (a,b,c) matches a scalar 3pt predicate it will be counted
             only once (b,a,c) for example, will not be counted.

             A compound predicate does not neglect redundant parameters.
             The reason for this is that in the general case with different
             thresholds for different pairs within the tuple, then even
             if (a,b,c) matches the predicte, (b,a,c) (for example) might or
             might not.

   SYMMETRY(new): The above statement only holds if you set the use_permute
                  argument to false on the command line.  The new code counts
                  all cases as in the "scalar" case above.
   */

--------------------------------------------------------

Other useful files in this directory....

vsimple.csv m1.ds t1.ds : Three tiny datafiles for demoes and testing

shortpaper.ps Alex Gray's and Andrew's short preliminary paper on this
              technology.

Understanding the source code:

   (1) Read www.cs.cmu.edu/~AUTON/programming.html
   (2) Read ../mrkd/mrkd.h
   (3) Read matcher.h
   (4) Read npt.c
