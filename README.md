# LSTM-acceleration-with-singular-value-decomposition
This is an extension of my work in [dncoble/LSTM-State-Estimation-with-Time-Domain-Signal](https://github.com/dncoble/LSTM-State-Estimation-with-Time-Domain-Signal) and [dncoble/LSTM-implementation-on-FPGA](https://github.com/dncoble/LSTM-implementation-on-FPGA). We use the SVD to reduce model size, assuming error will correspond to the singular values.

<p align="center">
<img src="./plots/reduce_rank.gif" alt="drawing" width="600"/>
</p>
<p align="center">
</p>


Although an LSTM cells are not linear, it contains four linear equations. Larger matrix equations both require more data storage and take more time to execute. The purpose of this project was to see if we could use singular value decompositions to reduce both data storage and increase computation time. This is done by changing matrix-vector multiplication into a two-step process. Both normal and reduced LSTM models are implemented using `tensorflow.keras` and linear algebra is performed with `numpy.linalg`. Timing with the python models is inconclusive, but I show that the amount of weights can be significantly reduced in size before error is adversely effected.


For more information, see the [attached powerpoint presentation](slides). If you have any questions, email me at dncoble@email.sc.edu
