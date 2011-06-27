#!/bin/bash

# Options
TEST_DIR="build/bin"
REPORT_DIR="reports"
TEST_OPTS="--report_level=detailed --log_level=test_suite --log_format=xml"
XML_REGEX="s/[^\x09\x0A\x0D\x20-\x{D7FF}\x{E000}-\x{FFFD}\x{10000}-\x{10FFFF}]//g"

echo "Wipe out old reports"
mkdir -p $REPORT_DIR
rm -rf $REPORT_DIR/*

echo "Finding test Binaries"
TEST_BINS=$(find $TEST_DIR -iname "*_test")

echo "Copy test_data_3_1000.csv to current working directory"
cp ./*/mlpack/neighbor_search/test_data_3_1000.csv .

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
   cat $BOOST_TEST | sed $XML_REGEX > $REPORT_DIR/boost_$BOOST_BASENAME
done

echo "Cleaning up working directory"
rm test_data*
