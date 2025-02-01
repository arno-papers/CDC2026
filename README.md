# CDC2026

Ideas:

1) Continue from
https://lirias.kuleuven.be/retrieve/709528
but instead of Kalman filter + Fisher information matrix, Particle filter + Entropy
https://arnostrouwen.com/posts/bayesian-experimental-design/
Running such an algorithm would be ultra expensive.
Needs really good GPU implementation.

2) Experimental design for Bayesian UDE, without the symbolic regression.
Estimating the parameters of the NN precisely is meaningless.
However predicting some function of the NN precisely is meaningful.
For the bioreactor example, there is some control function which causes maximal biomass to be produced.
But it is uncertain what this control is, due to missing physics.
An experiment is performed to reduce this uncertainty as much as possible.

3) Experimental design for Bayesian UDE, with SR
How can our DYCOPS idea be extended to Bayesian UDE?
