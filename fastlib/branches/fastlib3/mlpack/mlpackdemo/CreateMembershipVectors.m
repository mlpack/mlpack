function mv = CreateMembershipVectors(v)
mv = {};
for i = 1:size(v, 2)
    mv{i} = find(v(:, i) > 0);
end;
