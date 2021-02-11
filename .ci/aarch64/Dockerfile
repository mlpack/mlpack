FROM arm64v8/ubuntu
CMD ["-v","/home/vsts/work/1/s:/opt/share"]
RUN apt-get update
RUN apt-get install -y --allow-unauthenticated cmake tree
RUN ls /opt
RUN ls /
RUN uname -a
RUN tree /
RUN CTEST_OUTPUT_ON_FAILURE=1 ctest -T Test .
