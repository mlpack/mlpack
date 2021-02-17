#! /bash/sh
apt-get update
apt-get install -y --allow-unauthenticated cmake

cd /home/vsts/work/1/s/build
pwd

ls

CTEST_OUTPUT_ON_FAILURE=1 ctest -T Test .

