FROM arm64v8/ubuntu
RUN apt-get update
RUN apt-get install -y --allow-unauthenticated cmake tree
# ADD ../../ /home
# CMD ["-v","/home/vsts/work/1/s:/opt"]
# WORKDIR /home/vsts/work/1/s
COPY  . /
RUN cd /
RUN ls /
RUN ls /home/vsts/work/1/s
RUN tree /home
RUN pwd
RUN tar -xzvf mlpack.tar.gz
RUN tree /home
RUN cd /home/vsts/work/1/s/
RUN ls
RUN cd mlpack/build
RUN CTEST_OUTPUT_ON_FAILURE=1 ctest -T Test .
