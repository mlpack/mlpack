function []=plot_matrix
data=dlmread('scores.txt',' ');
data
%Plot all the columns of the matrix w.r.t the first column
bw=data(1:end,1:1);
plot_matrix=zeros(length(bw),3);
plot_matrix(1:end,2:2)=data(1:end,2:2);
plot_matrix(1:end,1:1)=data(1:end,3:3);
plot_matrix(1:end,3:3)=data(1:end,4:4);

plot(bw,plot_matrix);
legend('loc-poly','kde','likelihood');