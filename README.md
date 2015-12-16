# <img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/difacto.png width=130/> Distributed Factorization Machines

[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

Fast and memory efficient library for factorization machines (FM).

- Supports both ℓ1 regularized logistic regression and factorization
  machines.
- Runs on local machine and distributed clusters.
- Scales to datasets with billions examples and features.

### Quick Start

The following commands clone and build difacto, then download a sample dataset,
and train FM with 2-dimension on it.

```bash
git clone --recursive https://github.com/dmlc/difacto
make -j8
./tools/download.sh gisette data/
build/difacto data_in=data/gisette_scale val_data=data/gisette_scale.t \
lr=.02 V_dim=2 V_lr=.001
```

### History

Origins from
[wormhole/learn/difacto](https://github.com/dmlc/wormhole/tree/master/learn/difacto).

(NOTE: this project is still under developing, we hope to make the first release
at the end of 2015.)

### References

Mu Li, Ziqi Liu, Alex Smola, and Yu-Xiang Wang.
DiFacto — Distributed Factorization Machines. In WSDM, 2016
