FROM arm64v8/ubuntu
CMD ["-v","/home/vsts/work/1/s:/opt/share"]
RUN apt-get update
RUN apt-get install -y --allow-unauthenticated cmake
RUN ls /opt/share
RUN cd /opt/share/build && CTEST_OUTPUT_ON_FAILURE=1 ctest -T Test .
