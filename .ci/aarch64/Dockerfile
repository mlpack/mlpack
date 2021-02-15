FROM arm64v8/ubuntu
RUN apt-get update
RUN apt-get install -y --allow-unauthenticated cmake tree
COPY  . /
RUN cd /
RUN ls /
RUN tree /home
RUN pwd
RUN tar -xzvf mlpack.tar.gz
RUN tree /home
RUN cd /home/vsts/work/1/s/build && CTEST_OUTPUT_ON_FAILURE=1 ctest -T Test .

