# <img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/difacto.png width=130/> Distributed Factorization Machines

[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

Fast and memory efficient library for factorization machines (FM).

- Supports both ℓ1 regularized logistic regression and factorization
  machines.
- Runs on local machine and distributed clusters.
- Scales to datasets with billions examples and features.


### Quick Start

Building requires a C++ compiler supporting C++11, e.g. `gcc >= 4.8`.

1. Build difacto from source (with 8 threads) by `make -j8`

2. Download a sample dataset `./tools/download.sh gisette data/`

3. Run FM with 2-dimension

   ```bash
   build/difacto data_in=data/gisette_scale val_data=data/gisette_scale.t \
   lr=.02 V_dim=2 V_lr=.001
   ```

### History

Origins from
[wormhole/learn/difacto](https://github.com/dmlc/wormhole/tree/master/learn/difacto). But
this repository is still under developing.

### References

Mu Li, Ziqi Liu, Alex Smola, and Yu-Xiang Wang.
DiFacto — Distributed Factorization Machines. In WSDM, 2016
