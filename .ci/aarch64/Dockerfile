FROM arm64v8/ubuntu
RUN apt-get update
RUN apt-get install -y --allow-unauthenticated cmake
RUN cd /opt/share/build && CTEST_OUTPUT_ON_FAILURE=1 ctest -T Test .
