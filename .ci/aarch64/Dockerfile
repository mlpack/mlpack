FROM aaarch64/ubuntu
RUN apt-get update \
     apt-get install -y --allow-unauthenticated cmake \
     cd /opt/share/build && CTEST_OUTPUT_ON_FAILURE=1 ctest -T Test .
