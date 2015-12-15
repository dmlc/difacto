load data
%%
batch = 37
ix = [1 : 37 : 100, 101]

re = [];
for i = 1 : length(ix) - 1
  j = ix(i) : ix(i+1)-1;
  v = x(j,:);
  [a,b,c] = find(v');
  c(1:3)
  w = full(sparse(a,b,1));
  re = [re; [length(j), sum(y(j)), sum(cumsum(sum(w))), sum(abs(a)), sum(abs(b-1)), sum(c.*c)]];
end

for i = 1 : size(re, 2)
  re(:,i)'
end
