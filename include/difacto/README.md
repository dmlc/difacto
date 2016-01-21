This fold contains the abstract classes of difacto.

1. [Learner](learner.h) the base class of the learning algorithm. The system
   starts by calling `Learner.Run()`

2. [Loss](loss.h) the base class of a loss function, such as the logistic loss,
   which is able to evaluate the object value and calculate the gradients based
   on the weights and data.

3. [Updater](updater.h) the base class of a updater, which maintains the
   weights and allows to get (`Get()`) the weights and update (`Update()`) the
   weights based on the inputs (often the gradients)

4. [Store](store.h) the data communication interface for the Updater. So a remote
   node (such a worker) can get (`Pull()`) and update (`Push()`) the weights.

5. [Tracker](tracker.h) the control interface, which is able to
   issue remote procedure calls (RPCs) from the scheduler to any workers and
   servers, and monitors the progress.

The following figure shows the scheduler sends a RPC to a worker, which get and
update the weights on the servers.

 <img src=https://raw.githubusercontent.com/dmlc/web-data/master/difacto/class_arch.png width=500/>
