function [objv, gw, gV] = fm_loss_l2(y, X, w, X2, V, l2_w, l2_V)

[objv, gw, gV] = fm_loss(y, X, w, X2, V);
gw = gw + l2_w * w;
gV = gV + l2_V * V;
objv = objv + .5 * l2_w * sum(w(:).^2) + .5 * l2_V * sum(V(:).^2);
end
