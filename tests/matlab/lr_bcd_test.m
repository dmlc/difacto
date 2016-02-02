load data
a = sum(abs(x));
X = x(:,a~=0);
c
%%
load rcv1
y = Y;
%%

l1 = .1;
% l2 = .01;
lr = .05;
nblk = 1;

[n,p] = size(X);
w = zeros(p,1);
delta = ones(p,1);

blks = round(linspace(1,p+1,nblk+1))

for i = 1 : 11
  objv = sum(log(1 + exp ( - y .* (X * w))));
  fprintf('iter %d, objv %f, nnz w %d\n', i, objv, nnz(w));
  rdp = randperm(nblk);
  for b = 1 : nblk
    blk = false(p,1);
    blk(blks(rdp(b)) : blks(rdp(b)+1)-1) = true;

    tau = 1 ./ (1  + exp(y .* (X * w)));
    g = full(X(:,blk)' * (-y .* tau));
    h = full((X(:,blk).^2)' * (tau .* (1-tau))) / lr + 1e-6;

% soft-threadhold
    d = -w(blk);
    gp = g + l1;
    ix = gp <= h .* w(blk);
    d(ix) = - gp(ix) ./ h(ix);
    gn = g - l1;
    ix = gn >= h .* w(blk);
    d(ix) = - gn(ix) ./ h(ix);

    d = max(min(d, delta(blk)), -delta(blk));
    delta(blk) = 2*abs(d) + .1;
    w(blk) = w(blk) + d;
    fprintf('%f %f %f %f\n', norm(g)^2, norm(h*lr)^2, norm(w)^2, norm(delta)^2);
  end

  delta = max(min(delta, 5), -5);
end
