num_ics = size(ic_scores, 1);
neg_threshold = -2;
pos_threshold = 2;

neg_counts = zeros(num_ics, 9);
pos_counts = zeros(num_ics, 9);
for i = 1:num_ics
  neg_counts(i,:) = hist(clusters(find(ic_scores(i,:) < neg_threshold)), -1:7);
end

pos_counts = zeros(num_ics, 9);
for i = 1:num_ics
  pos_counts(i,:) = hist(clusters(find(ic_scores(i,:) > pos_threshold)), -1:7);
end