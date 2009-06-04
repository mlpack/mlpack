function [] = plot_kpcs(kpc, comp1, comp2)
% function [] = plot_kpcs(comp1, comp2)

ranges = cell(5,1);
ranges{1} = 1:4;
ranges{2} = 5:18;
ranges{3} = 19:60;
ranges{4} = 61:74;
ranges{5} = 75:128;

for i = 1:5
  ranges{i} = setdiff(ranges{i}, [19 39 77]);
end

colors  = 'kbrmg';
markers = '*+xo^';
clf;
hold on;
for i = 1:5
  scatter(kpc(ranges{i},comp1), ...
	  kpc(ranges{i},comp2), ...
	  [colors(i) markers(i)]);
  %scatter(kpc(ranges{i},comp1), zeros(length(ranges{i}), 1), colors(i));
end

%scatter(kpc(19,comp1), kpc(19, comp2), 'rv');
%scatter(kpc(39,comp1), kpc(39, comp2), 'rx');
%scatter(kpc(77,comp1), kpc(77, comp2), 'gx');