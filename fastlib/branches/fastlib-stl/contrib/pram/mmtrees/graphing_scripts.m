% OPTDIGITS
% validated error graphs

% Max-margin trees, variation within C:

axes('fontsize', 14, 'TickDir', 'out');
hold;
for i = 1:5
tmp = res_lmtree_rann_all(i).nn_res(:,1) + res_lmtree_rann_all(i).nn_res(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_lmtree_rann_all(i).nn_res(I, 3) * 100 / 1347, 'Color', res_lmtree_rann_all(i).color);
end
legend('C = 0.001', 'C = 0.005', 'C = 0.01', 'C = 0.05', 'C = 0.1');
title('Performance with varying C', 'fontsize', 16);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Average rank error (in %-tile)', 'fontsize', 16);
axis([0 700 0 7]);

axes('fontsize', 14, 'TickDir', 'out'); hold;
for i = 1:5
tmp = res_lmtree_rann_all(i).nn_res(:,1) + res_lmtree_rann_all(i).nn_res(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_lmtree_rann_all(i).nn_res(I, 4) * 100 / 1347, 'Color', res_lmtree_rann_all(i).color);
end
legend('C = 0.001', 'C = 0.005', 'C = 0.01', 'C = 0.05', 'C = 0.1');
title('Performance with varying C', 'fontsize', 16);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 800 0 42]);


% RPtrees, variation within c:

axes('fontsize', 14, 'TickDir', 'out'); hold;
for i = 1:length(c)
tmp = res_c(i).a(:,1) + res_c(i).a(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_c(i).a(I, 3) * 100 / 1347, 'Color', res_c(i).color);
end
legend('c = 1.0', 'c = 1.2', 'c = 1.5', 'c = 1.75', 'c = 2.0', 'c = 2.5', 'c = 3.0', 'c = 3.5');
title('Performance with varying c', 'fontsize', 16);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Average rank error (%-tile)', 'fontsize', 16);
axis([0 700 0 7]);


axes('fontsize', 14, 'TickDir', 'out');
hold;
for i = 1:length(c)
tmp = res_c(i).a(:,1) + res_c(i).a(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_c(i).a(I, 4) * 100 / 1347, 'Color', res_c(i).color);
end
legend('c = 1.0', 'c = 1.2', 'c = 1.5', 'c = 1.75', 'c = 2.0', 'c = 2.5', 'c = 3.0', 'c = 3.5');
title('Performance with varying c', 'fontsize', 16);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 800 0 42]);

% All trees:

axes('fontsize', 14, 'TickDir', 'out'); hold;
tmp = res_lmtree_rann_all(3).nn_res(:,1) + res_lmtree_rann_all(3).nn_res(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_lmtree_rann_all(3).nn_res(I,3) * 100 / 1347, '-r', 'Linewidth', 2);
[S, I] = sort(res_kdtree_rann(:,1));
plot(res_kdtree_rann(I,1), res_kdtree_rann(I,2) * 100 / 1347, '--k', 'Linewidth', 2);
tmp = res_c(3).a(:,1) + res_c(3).a(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_c(3).a(I, 3) * 100 / 1347, '-.c', 'Linewidth', 2);
legend('MM-tree', 'KD-tree', 'RP-tree');
title('Error Constrained NN', 'fontsize', 17);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Average rank error (in %-tile)', 'fontsize', 16);
axis([0 800 0 7]);

axes('fontsize', 14, 'TickDir', 'out'); hold;
tmp = res_lmtree_rann_all(3).nn_res(:,1) + res_lmtree_rann_all(3).nn_res(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_lmtree_rann_all(3).nn_res(I,4) * 100 / 1347, '-r', 'Linewidth', 2);
[S, I] = sort(res_kdtree_rann(:,1));
plot(res_kdtree_rann(I,1), res_kdtree_rann(I,3) * 100 / 1347, '--k', 'Linewidth', 2);
tmp = res_c(3).a(:,1) + res_c(3).a(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_c(3).a(I,4) * 100 / 1347, '-.c', 'Linewidth', 2);
legend('MM-tree', 'KD-tree', 'RP-tree');
title('Error constrained NN', 'fontsize', 17);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 800 0 42]);


% time constrained graphs

% Max-margin trees, variation within C:

axes('fontsize', 14, 'TickDir', 'out'); hold;
for i = 2:6
plot((1:length(ann_accuracy(i).max)), ann_accuracy(i).max * 100 / 1347, '-', 'Color', ann_accuracy(i).color);
end
legend('C = 0.001', 'C = 0.005', 'C = 0.01', 'C = 0.05', 'C = 0.1');
title('Performance with varying C', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 40 0 30]);


axes('fontsize', 14, 'TickDir', 'out'); hold;
for i = 2:6
plot((1:length(ann_accuracy(i).means)), ann_accuracy(i).means * 100 ...
     / 1347, '-', 'Color', ann_accuracy(i).color);
end
legend('C = 0.001', 'C = 0.005', 'C = 0.01', 'C = 0.05', 'C = 0.1');
title('Performance with varying C', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Average rank error (in %-tile)', 'fontsize', 16);
axis([0 40 0 0.6]);

%axes('fontsize', 14, 'TickDir', 'out'); hold;
%for i = 2:6
%plot((1:length(ann_accuracy(i).means)), ann_accuracy(i).means, '-', 'Color', ann_accuracy(i).color);
%plot((1:length(ann_accuracy(i).means)), ann_accuracy(i).means + 2*ann_accuracy(i).stds, '-.', 'Color', ann_accuracy(i).color);
%end
%legend('C = 0.001 : Mean', 'C = 0.001 : Mean + 2*std', 'C = 0.005 : Mean', 'C = 0.005 : Mean + 2*std','C = 0.01 : Mean', 'C = 0.01 : Mean + 2*std','C = 0.05 : Mean', 'C = 0.05 : Mean + 2*std','C = 0.1 : Mean', 'C = 0.1 : Mean + 2*std');
%title('Performance with varying values of C', 'fontsize', 18);
%xlabel('Number of leaves visited', 'fontsize', 16);
%ylabel('Average rank error', 'fontsize', 16);


% RP-trees variation within c:

axes('fontsize', 14, 'TickDir', 'out'); hold;
for i = 1:8
plot((1:length(ann_acc_RPTree(i).a.means)), ann_acc_RPTree(i).a.means ...
     * 100 / 1347, '-', 'Color', res_c(i).color);
end
legend('c = 1.0', 'c = 1.2', 'c = 1.5', 'c = 1.75', 'c = 2.0', 'c = 2.5', 'c = 3.0', 'c = 3.5');
title('Performance with varying c', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Average rank error (in %-tile)', 'fontsize', 16);
axis([0 40 0 0.6]);

axes('fontsize', 14, 'TickDir', 'out'); hold;
for i = 1:8
plot((1:length(ann_acc_RPTree(i).a.max)), ann_acc_RPTree(i).a.max * ...
     100 / 1347, '-', 'Color', res_c(i).color);
end
legend('c = 1.0', 'c = 1.2', 'c = 1.5', 'c = 1.75', 'c = 2.0', 'c = 2.5', 'c = 3.0', 'c = 3.5');
title('Performance with varying c', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 40 0 30]);


% All trees:

axes('fontsize', 14, 'TickDir', 'out'); hold;
plot((1:length(ann_accuracy(4).means)), ann_accuracy(4).means * 100 ...
     / 1347, '-r', 'Linewidth', 2);
plot((1:length(ann_accuracy(4).means)), (ann_accuracy(4).means + ...
					 ann_accuracy(4).stds) * 100 ...
     / 1347, '-.r', 'Linewidth', 2);
plot((1:length(ann_acc_KDTree(:,1))), ann_acc_KDTree(:,1) * 100 ...
     / 1347, '-k', 'Linewidth', 2);
plot((1:length(ann_acc_KDTree(:,1))), (ann_acc_KDTree(:,1) + ...
				       ann_acc_KDTree(:,2)) * 100 / ...
     1347, '-.k', 'Linewidth', 2);
plot((1:length(ann_acc_RPTree(7).a.means)), ann_acc_RPTree(7).a.means ...
     * 100 / 1347, '-c', 'Linewidth', 2);
plot((1:length(ann_acc_RPTree(7).a.means)), (ann_acc_RPTree(7).a.means ...
					     + ann_acc_RPTree(7).a.stds) ...
     * 100 / 1347, '-.c', 'Linewidth', 2);
legend('MM-tree: mean', 'MM-tree: mean + std', 'KD-tree: mean', 'KD-tree: mean + std', 'RP-tree: mean', 'RP-tree: mean + std');
title('Time constrained NN', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Rank error (in %-tile)', 'fontsize', 16);
axis([0 50 0 1.5]);


%axes('fontsize', 14, 'TickDir', 'out'); hold;
%plot((1:length(ann_accuracy(4).means)), ann_accuracy(4).means, '-r', 'Linewidth', 2);
%plot((1:length(ann_acc_KDTree(:,1))), ann_acc_KDTree(:,1), '--k', 'Linewidth', 2);
%plot((1:length(ann_acc_RPTree(7).a.means)), ann_acc_RPTree(7).a.means, '-.b', 'Linewidth', 2);
%legend('MM-tree', 'KD-tree', 'RP-tree');
%title('Time constrained NN: different tree performance', 'fontsize', 17);
%xlabel('Number of leaves visited', 'fontsize', 16);
%ylabel('Average rank error', 'fontsize', 16);



axes('fontsize', 14, 'TickDir', 'out'); hold;
plot((1:length(ann_accuracy(4).means)), ann_accuracy(4).max * 100 / ...
     1347, '-r', 'Linewidth', 2);
plot((1:length(ann_acc_KDTree(:,1))), ann_acc_KDTree(:,4) * 100 / ...
     1347, '-k', 'Linewidth', 2);
plot((1:length(ann_acc_RPTree(7).a.means)), ann_acc_RPTree(7).a.max ...
     * 100 / 1347, '-c', 'Linewidth', 2);
legend('MM-tree', 'KD-tree', 'RP-tree');
title('Time constrained NN', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 50 0 15]);


% MNist Validated error graphs:

% RPTrees

axes('fontsize', 14, 'TickDir', 'out');hold;
for i = 1:length(c)
tmp = res_RPTree2_c_mnist(i).a(:,1) + res_RPTree2_c_mnist(i).a(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_RPTree2_c_mnist(i).a(I, 3) * 100 / 54000,...
     'Color', res_c(i).color);
end
legend('c = 1.0', 'c = 1.2', 'c = 1.5', 'c = 1.75', 'c = 2.0', 'c = 2.5', 'c = 3.0', 'c = 3.5');
title('Performance with varying c', 'fontsize', 16);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Average rank error (in %-tile)', 'fontsize', 16);
axis([0 2500 0 7]);


axes('fontsize', 14, 'TickDir', 'out');hold;
for i = 1:length(c)
tmp = res_RPTree2_c_mnist(i).a(:,1) + res_RPTree2_c_mnist(i).a(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_RPTree2_c_mnist(i).a(I, 4) * 100 / 54000,...
     'Color', res_c(i).color);
end
legend('c = 1.0', 'c = 1.2', 'c = 1.5', 'c = 1.75', 'c = 2.0', 'c = 2.5', 'c = 3.0', 'c = 3.5');
title('Performance with varying c', 'fontsize', 16);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 2500 0 50]);


% MMtrees
axes('fontsize', 14, 'TickDir', 'out');hold;
for i = 1:8
tmp = res_lmtree_mnist_rann_all(i).nn_res(:,1) + res_lmtree_mnist_rann_all(i).nn_res(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_lmtree_mnist_rann_all(i).nn_res(I, 3) * 100 / 54000,...
     'Color', res_c(i).color);
end
legend('C = 0.000001', 'C = 0.000005', 'C = 0.00001', 'C = 0.00005', 'C = 0.0001', 'C = 0.0005', 'C = 0.001', 'C = 0.005');
title('Performance with varying C', 'fontsize', 16);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Average rank error (in %-tile)', 'fontsize', 16);
axis([0 2500 0 7]);


axes('fontsize', 14, 'TickDir', 'out');hold;
for i = 1:8
tmp = res_lmtree_mnist_rann_all(i).nn_res(:,1) + res_lmtree_mnist_rann_all(i).nn_res(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_lmtree_mnist_rann_all(i).nn_res(I, 4) * 100 / 54000,...
     'Color', res_c(i).color);
end
legend('C = 0.000001', 'C = 0.000005', 'C = 0.00001', 'C = 0.00005', 'C = 0.0001', 'C = 0.0005', 'C = 0.001', 'C = 0.005');
title('Performance with varying C', 'fontsize', 16);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 2500 0 50]);



% All trees
axes('fontsize', 14, 'TickDir', 'out'); hold;
tmp = res_lmtree_mnist_rann_all(3).nn_res(:,1) + res_lmtree_mnist_rann_all(3).nn_res(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_lmtree_mnist_rann_all(3).nn_res(I,3) * 100 / 54000,...
     '-r', 'Linewidth', 2);
[S, I] = sort(res_kdtree_mn_rann(:,1));
plot(res_kdtree_mn_rann(I,1), res_kdtree_mn_rann(I,2) * 100 / 54000,...
     '--k', 'Linewidth', 2);
tmp = res_RPTree2_c_mnist(8).a(:,1) + res_RPTree2_c_mnist(8).a(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_RPTree2_c_mnist(8).a(I, 3) * 100 / 54000,...
     '-.c', 'Linewidth', 2);
legend('MM-tree', 'KD-tree', 'RP-tree');
title('Error constrained NN', 'fontsize', 16);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Average rank error (in %-tile)', 'fontsize', 16);
axis([0 2500 0 7]);

axes('fontsize', 14, 'TickDir', 'out'); hold;
tmp = res_lmtree_mnist_rann_all(3).nn_res(:,1) + res_lmtree_mnist_rann_all(3).nn_res(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_lmtree_mnist_rann_all(3).nn_res(I,4) * 100 / 54000,...
     '-r', 'Linewidth', 2);
[S, I] = sort(res_kdtree_mn_rann(:,1));
plot(res_kdtree_mn_rann(I,1), res_kdtree_mn_rann(I,3) * 100 / 54000,...
     '--k', 'Linewidth', 2);
tmp = res_RPTree2_c_mnist(8).a(:,1) + res_RPTree2_c_mnist(8).a(:,2);
[S, I] = sort(tmp);
plot(tmp(I), res_RPTree2_c_mnist(8).a(I, 4) * 100 / 54000,...
     '-.c', 'Linewidth', 2);
legend('MM-tree', 'KD-tree', 'RP-tree');
title('Error constrained NN', 'fontsize', 16);
xlabel('Average number of distance computations', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 2500 0 50]);

% MNist time-constrained error: 

% RPTrees
axes('fontsize', 14, 'TickDir', 'out'); hold;
for i = 1:8
plot((1:length(ann_acc_RPTree2_mnist(i).a.means)),...
     ann_acc_RPTree2_mnist(i).a.means * 100 / 54000,...
     '-', 'Color', res_c(i).color);
end
legend('c = 1.0', 'c = 1.2', 'c = 1.5', 'c = 1.75', 'c = 2.0', 'c = 2.5', 'c = 3.0', 'c = 3.5');
title('Performance with varying c', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Average rank error (in %-tile)', 'fontsize', 16);
axis([0 200 0 4]);


axes('fontsize', 14, 'TickDir', 'out'); hold;
for i = 1:8
plot((1:length(ann_acc_RPTree2_mnist(i).a.max)), ...
     ann_acc_RPTree2_mnist(i).a.max * 100 / 54000,...
     '-', 'Color', res_c(i).color);
end
legend('c = 1.0', 'c = 1.2', 'c = 1.5', 'c = 1.75', 'c = 2.0', 'c = 2.5', 'c = 3.0', 'c = 3.5');
title('Performance with varying c', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 200 0 40]);

% MMTrees
axes('fontsize', 14, 'TickDir', 'out'); hold;
for i = 1:8
plot((1:length(ann_acc_mnistLMTree(i).means)), ...
     ann_acc_mnistLMTree(i).means * 100 / 54000,...
     '-', 'Color', res_c(i).color);
end
legend('C = 0.000001', 'C = 0.000005', 'C = 0.00001', 'C = 0.00005', 'C = 0.0001', 'C = 0.0005', 'C = 0.001', 'C = 0.005');
title('Performance with varying C', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Average rank error (in %-tile)', 'fontsize', 16);
axis([0 200 0 0.35]);


axes('fontsize', 14, 'TickDir', 'out'); hold;
for i = 1:8
plot((1:length(ann_acc_mnistLMTree(i).max)), ...
     ann_acc_mnistLMTree(i).max * 100 / 54000,...
     '-', 'Color', res_c(i).color);
end
legend('C = 0.000001', 'C = 0.000005', 'C = 0.00001', 'C = 0.00005', 'C = 0.0001', 'C = 0.0005', 'C = 0.001', 'C = 0.005');
title('Performance with varying C', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 200 0 40]);


% All trees

%axes('fontsize', 14, 'TickDir', 'out'); hold;
%plot((1:length(ann_acc_mnistLMTree(3).means)), ann_acc_mnistLMTree(3).means, '-r', 'Linewidth', 2);
%plot((1:length(ann_acc_KDtree_mnist(:,1))), ann_acc_KDtree_mnist(:,1), '--k', 'Linewidth', 2);
%plot((1:length(ann_acc_RPTree2_mnist(8).a.means)), ann_acc_RPTree2_mnist(8).a.means, '-.b', 'Linewidth', 2);
%legend('MM-tree', 'KD-tree', 'RP-tree');
%title('Time constrained NN: different tree performance', 'fontsize', 17);
%xlabel('Number of leaves visited', 'fontsize', 16);
%ylabel('Average rank error', 'fontsize', 16);
%axis([0 3000 0 600]);
%axis([0 1000 0 400]);
%axis([0 600 0 300]);

axes('fontsize', 14, 'TickDir', 'out'); hold;
plot((1:length(ann_acc_mnistLMTree(3).max)), ...
     ann_acc_mnistLMTree(3).max * 100 / 54000,...
     '-r', 'Linewidth', 2);
plot((1:length(ann_acc_KDtree_mnist(:,1))), ...
     ann_acc_KDtree_mnist(:,4) * 100 / 54000,...
     '-k', 'Linewidth', 2);
plot((1:length(ann_acc_RPTree2_mnist(8).a.max)),...
     ann_acc_RPTree2_mnist(8).a.max * 100 / 54000,...
     '-c', 'Linewidth', 2);
legend('MM-tree', 'KD-tree', 'RP-tree');
title('Time constrained NN', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 200 0 40]);


%axis([0 100 0 28000]); % x-zoom
%axis([0 1000 0 10000]); % y-zoom
%axis([0 2500 0 10000]);
%axis([0 2500 0 6000]);

axes('fontsize', 14, 'TickDir', 'out'); hold;
plot((1:length(ann_acc_mnistLMTree(3).means)), ...
     ann_acc_mnistLMTree(3).means * 100 / 54000, ...
     '-r', 'Linewidth', 2);
plot((1:length(ann_acc_mnistLMTree(3).means)), ...
     (ann_acc_mnistLMTree(3).means + ann_acc_mnistLMTree(3).stds) * ...
     100 / 54000, '-.r', 'Linewidth', 2);
plot((1:length(ann_acc_KDtree_mnist(:,1))), ...
     ann_acc_KDtree_mnist(:,1) * 100 / 54000,...
     '-k', 'Linewidth', 2);
plot((1:length(ann_acc_KDtree_mnist(:,1))), ...
     (ann_acc_KDtree_mnist(:,1) + ann_acc_KDtree_mnist(:,2)) * 100 ...
     / 54000, '-.k', 'Linewidth', 2);
plot((1:length(ann_acc_RPTree2_mnist(8).a.means)),...
     ann_acc_RPTree2_mnist(8).a.means * 100 / 54000, ...
     '-c', 'Linewidth', 2);
plot((1:length(ann_acc_RPTree2_mnist(8).a.means)),...
     (ann_acc_RPTree2_mnist(8).a.means + ann_acc_RPTree2_mnist(8).a.stds) ...
     * 100 / 54000, '-.c', 'Linewidth', 2);
legend('MM-tree: mean', 'MM-tree: mean + std', 'KD-tree: mean', 'KD-tree: mean + std', 'RP-tree: mean', 'RP-tree: mean + std');
title('Time constrained NN', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Rank error (%-tile)', 'fontsize', 16);
axis([0 200 0 1]);

%axis([0 2500 0 1000]);
%axis([0 1000 0 400]);
%axis([0 500 0 200]); % y-zoom
%axis([0 100 0 4000]); % x-zoom




% Images data -- time constrained NN

axes('fontsize', 14, 'TickDir', 'out'); hold;
plot((1:length(ann_acc_imagesLMTree(6).means)), ...
     ann_acc_imagesLMTree(6).means * 100 / 523,...
     '-r', 'Linewidth', 2);
plot((1:length(ann_acc_imagesLMTree(6).means)), ...
     (ann_acc_imagesLMTree(6).means + ann_acc_imagesLMTree(6).stds) ...
     * 100 / 523, '-.r', 'Linewidth', 2);
plot((1:length(ann_acc_KDtree_images(:,1))), ...
     ann_acc_KDtree_images(:,1) * 100 / 523,...
     '-k', 'Linewidth', 2);
plot((1:length(ann_acc_KDtree_images(:,1))), ...
     (ann_acc_KDtree_images(:,1) + ann_acc_KDtree_images(:,2))...
     * 100 / 523, '-.k', 'Linewidth', 2);
plot((1:length(ann_acc_RPTree2_images(5).a.means)), ...
     ann_acc_RPTree2_images(5).a.means * 100 / 523,...
     '-c', 'Linewidth', 2);
plot((1:length(ann_acc_RPTree2_images(5).a.means)),...
     (ann_acc_RPTree2_images(5).a.means + ann_acc_RPTree2_images(5).a.stds) ...
     * 100 / 523, '-.c', 'Linewidth', 2);
legend('MM-tree: mean', 'MM-tree: mean + std', 'KD-tree: mean', 'KD-tree: mean + std', 'RP-tree: mean', 'RP-tree: mean + std');
title('Time constrained NN', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Rank error (in %-tile)', 'fontsize', 16);
axis([0.9 10 0 3]);

axes('fontsize', 14, 'TickDir', 'out'); hold;
plot((1:length(ann_acc_imagesLMTree(6).max)), ...
     ann_acc_imagesLMTree(6).max * 100 / 523, ...
     '-r', 'Linewidth', 2);
plot((1:length(ann_acc_KDtree_images(:,1))), ...
     ann_acc_KDtree_images(:,4) * 100 / 523, ...
     '-k', 'Linewidth', 2);
plot((1:length(ann_acc_RPTree2_images(5).a.max)),...
     ann_acc_RPTree2_images(5).a.max * 100 / 523, ...
     '-c', 'Linewidth', 2);
legend('MM-tree', 'KD-tree', 'RP-tree');
title('Time constrained NN', 'fontsize', 17);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0.9 10 0 20]);



% Physics data -- time constrained NN
% C = 0.1;

axes('fontsize', 14, 'TickDir', 'out'); hold;

plot((1:length(TC_mmtree_phy.ann_c(8).a.means)), ...
     TC_mmtree_phy.ann_c(8).a.means * 100 / 112500,...
     '-r', 'LineWidth', 2);
plot((1:length(TC_mmtree_phy.ann_c(8).a.means)), ...
     (TC_mmtree_phy.ann_c(8).a.means + ...
      TC_mmtree_phy.ann_c(8).a.stds) * 100 / 112500,...
     '-.r', 'LineWidth', 2);

plot((1:length(TC_kdtree_phy(:,1))),...
     TC_kdtree_phy(:,1) * 100 / 112500,...
     '-k', 'LineWidth', 2);
plot((1:length(TC_kdtree_phy(:,1))), ...
     (TC_kdtree_phy(:,1) + TC_kdtree_phy(:,2)) * 100 / 112500,...
     '-.k', 'LineWidth', 2);

plot((1:length(TC_rptree_phy.ann_c(2).a.means)), ...
     TC_rptree_phy.ann_c(2).a.means * 100 / 112500,...
     '-c', 'LineWidth', 2);
plot((1:length(TC_rptree_phy.ann_c(2).a.means)), ...
     (TC_rptree_phy.ann_c(2).a.means + ...
      TC_rptree_phy.ann_c(2).a.stds) * 100 / 112500,...
     '-.c' , 'LineWidth', 2);

legend('MM-tree: mean', 'MM-tree: mean + std', 'KD-tree: mean', 'KD-tree: mean + std', 'RP-tree: mean', 'RP-tree: mean + std');
title('Time constrained NN', 'fontsize', 16);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Rank error (in %-tile)', 'fontsize', 16);
axis([0 100 0 0.05]);

axes('fontsize', 14, 'TickDir', 'out'); hold;
plot((1:length(TC_mmtree_phy.ann_c(8).a.max)), ...
     TC_mmtree_phy.ann_c(8).a.max * 100 / 112500,...
     '-r', 'LineWidth', 2);
plot((1:length(TC_kdtree_phy(:,1))),...
     TC_kdtree_phy(:,4) * 100 / 112500,...
     '-k', 'LineWidth', 2);
plot((1:length(TC_rptree_phy.ann_c(2).a.max)), ...
     TC_rptree_phy.ann_c(2).a.max * 100 / 112500,...
     '-c', 'LineWidth', 2);
legend('MM-tree', 'KD-tree', 'RP-tree');
title('Time constrained NN', 'fontsize', 17);
xlabel('Number of leaves visited', 'fontsize', 16);
ylabel('Maximum rank error (in %-tile)', 'fontsize', 16);
axis([0 100 0 25]);


