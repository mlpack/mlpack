FROM arm64v8/ubuntu
CMD ["-v","/home/vsts/work/1/s:/opt"]
RUN apt-get update
RUN apt-get install -y --allow-unauthenticated cmake tree
RUN ls /opt
RUN cd /opt
RUN ls .
RUN uname -a
RUN CTEST_OUTPUT_ON_FAILURE=1 ctest -T Test .
