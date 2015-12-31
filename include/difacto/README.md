This folds contains the abstract classes of difacto.

1. [Learner](learner.h) the base class of the learning algorithm. The system
   starts by calling Learner.Run()

2. [Loss](loss.h) the base class of a loss function, such as logisitic loss,
   which is able to evaluate the object value and calculate the gradients based
   on the weights and data.

3. [Updater](updater.h) the base class of a updater, which maintains the
   weights and supports to get (`Get()`) the weights and update (`Update()`) the
   weights based on the input (often the gradients)

4. [Store](store.h) the data communication interface for the Updater. So a remote
   node can get (`Pull()`) and update (`Push()`) the weights.

5. [Tracker](tracker.h) the control communication interface, which is able to
   issue remote procedure calls (RPCs) from the scheduler to any worker and
   server, and monitor the progress

A example interactive diagram between the scheduler, a worker, and the servers:

 <img src=https://raw.githubusercontent.com/dmlc/web-data/master/difacto/class_arch.png width=600/>
