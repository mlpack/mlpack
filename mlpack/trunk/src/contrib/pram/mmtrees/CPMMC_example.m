% This is an example on how to use the Cutting Plane Maximum Margin
% Clustering (CPMMC) code.
% 
% Author: Bin Zhao
% Created on: 19 April, 2008
% Last Updated on : 19 April, 2008


%clear all;

% Load UCI digits data set
%load('optdigits.txt');
A=optdigits';
tempLabel=A(65,:);
A(65,:)=[];

% We solve the binary clustering problem, choose classes '1' and '7' for an
% example
classlabel1 = 1;
classlabel2 = 7;

indexLabel = find(tempLabel==classlabel1|tempLabel==classlabel2);
MyData = A(:,indexLabel);
%display(size(MyData));
correctLabel = tempLabel(indexLabel);
indexLabel1 = find(correctLabel == classlabel1);
correctLabel(indexLabel1) = -1;
indexLabel2 = find(correctLabel == classlabel2);
correctLabel(indexLabel2) = 1;

[Acc timeElapsed] = CPMMC2(MyData,correctLabel);
