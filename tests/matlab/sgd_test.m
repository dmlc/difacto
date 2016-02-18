%%
load data
[i,j,k]=find(x);
cnt = full(sum(sparse(i,j,ones(size(i)))));
X = x(:,cnt>0);
Y = y;
X2 = x(:,cnt>0);

%%
lr = 1;
lr_V = .8;
l1 = 1;
l2 = .1;
V_l2 = .1;

V_dim = 0;
if V_dim == 0
  X2 = [];
end

[n, p] = size(X);
w = zeros(p,1);
V = repmat(((1:V_dim) - V_dim/2)*.01, size(X2,2), 1);

sq_w = zeros(size(w));
sq_V = zeros(size(V));
z = zeros(size(w));

for k = 1 : 20
  [objv, gw, gV] = fm_loss_l2(Y, X, w, X2, V, lr, lr_V);
  objv = objv + l1 * sum(abs(w));
  objv = fm_loss(Y, X, w, X2, V);
  sq_w_new = sqrt(sq_w.*sq_w + gw.*gw);
  sq_V = sqrt(sq_V.*sq_V + gV.*gV);

  z = z - gw - (sq_w - sq_w_new) ./ lr .* w;
  sq_w = sq_w_new;

  ix = (z <= l1) & (z >= -l1);
  w(ix) = 0;
  eta = (1 + sq_w) / lr;

  ix1 = (~ ix) & (z > 0);
  w(ix1) = (z(ix1) - l1) ./ eta(ix1);

  ix2 = (~ ix) & (z < 0);
  w(ix2) = (z(ix2) + l1) ./ eta(ix2);

  V = V - lr_V * gV./(sq_V+1);

  fprintf('objv = %f, nnz_w = %f\n', objv, nnz(w))
end
