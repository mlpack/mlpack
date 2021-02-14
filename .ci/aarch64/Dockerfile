FROM arm64v8/ubuntu
# ADD ../../ /home
# CMD ["-v","/home/vsts/work/1/s:/opt"]
WORKDIR /home/vsts/work/1/s
COPY  . /home/vsts/work/1/s/
RUN apt-get update
RUN apt-get install -y --allow-unauthenticated cmake tree
RUN tree /home
RUN tar -xzvf mlpack.tar.gz
RUN cd /home/vsts/work/1/s/
RUN ls
RUN cd mlpack/build
RUN CTEST_OUTPUT_ON_FAILURE=1 ctest -T Test .
