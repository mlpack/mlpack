#!/bin/bash

# Options
TEST_DIR="build/bin"
REPORT_DIR="reports/tests"
TEST_OPTS="--report_level=detailed --log_level=test_suite --log_format=xml"
XML_REGEX="[:print:]"

echo "Wipe out old reports"
mkdir -p $REPORT_DIR
rm -rf $REPORT_DIR/*

echo "Finding test Binaries"
TEST_BINS=$(find $TEST_DIR -iname "*_test")

echo "Copy test_data_3_1000.csv to current working directory"
cp ./*/src/mlpack/methods/neighbor_search/test_data_3_1000.csv .

echo "Copy ARFF files for NBC test to current working directory"
cp ./*/src/mlpack/methods/naive_bayes/*.arff .

echo "Running All Tests:"
for ML_TEST in ${TEST_BINS}
do
   echo "[$(basename $(pwd))] $ML_TEST"
   ./$ML_TEST $TEST_OPTS > $REPORT_DIR/$(basename $ML_TEST).xml
done

echo "Finding Boost.Test Results:"
BOOST_RESULTS=$(grep -l "<TestLog>" $REPORT_DIR/*)
for BOOST_TEST in ${BOOST_RESULTS}
do
   BOOST_BASENAME=$(basename $BOOST_TEST)
   echo "Found boost_$BOOST_BASENAME"
   #mv $BOOST_TEST $REPORT_DIR/boost_$BOOST_BASENAME
   cat $BOOST_TEST | tr -cd  $XML_REGEX > $REPORT_DIR/boost_$BOOST_BASENAME
done

echo "Cleaning up working directory"
rm test_data*
