FROM arm64v8/ubuntu
ADD /home/vsts/work/1/s /home/vsts/work/1/s
#CMD ["-v","/home/vsts/work/1/s:/opt"]
RUN apt-get update
RUN apt-get install -y --allow-unauthenticated cmake tree
RUN ls /home
RUN cd /home
RUN ls .
RUN uname -a
RUN CTEST_OUTPUT_ON_FAILURE=1 ctest -T Test .
