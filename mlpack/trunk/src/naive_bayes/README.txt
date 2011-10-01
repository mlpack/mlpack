Author : Parikshit Ram (pram@cc.gatech.edu)

The files are the following:
1. nbc_main.cc - this is the main which creates an object of the class SimpleNaiveBayesClassifier, trains it, tests it and outputs the results. The executable formed is called "nbc".
 - the parameters taken in by main are the following:
   --train : the file that contains the training data, the last column being the class of the data point
   --nbc/classes : the number of classes the data provided has been classified into
   --test : this file contains the testing data, this still contains its actual labels on the last column, but it is not used.
   --output : the file into which you want the output to be written into, defaults to "output.csv"

2. simple_nbc.h - this is the file that contains the definition of the class SimpleNaiveBayesClassifier. The rest of the details are present in the file itself.

3. phi.h - this contains the functions that calculate the value of the univariate and multivariate Gaussian probability density function

4. test_simple_nbc_main.cc - this file contains the class which tests the class SimpleNaiveBayesClassifier. The executable formed is "nbc_test".
 - this takes no parameters

5. the .arff files, whose use has been described above.

-> An example run would the following:
fl-build nbc_main
./nbc_main --train=trainSet.arff  --nbc/classes=2 --test=testSet.arff --output=output_example.csv

-> An example run of the testing class would be the following:
fl-build test_simple_nbc_main
./nbc_test

Note: You don't need to give any parameters for testing because it will use the defaults. It requires the files be in the same directory as you run the executable from.
