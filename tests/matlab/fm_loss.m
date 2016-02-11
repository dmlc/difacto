function [objv, gw, gV] = fm_loss(y, X, w, X2, V)
% X, w for the linear term
% X2, V for the embedding term

py = X * w;
if ~isempty(X2)
  py = py + .5 * sum((X2*V).^2 - (X2.*X2)*(V.*V), 2);
end

objv = sum(log(1+exp(-y .* py)));
p = - y ./ (1 + exp (y .* py));
gw = X' * p;

if ~isempty(X2)
  gV = X2' * bsxfun(@times, p, X2*V) - bsxfun(@times, (X2.*X2)'*p, V);
else
  gV = [];
end


end
