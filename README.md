# send-doods

### About
Generative Adversarial Network implementation.

### Dependencies
python, numpy, scipy, tensorflow (gpu version recommended).

### Running
`python main.py`

arguments <br />
`--data_dir <directory for storing input data>` <br />
`--use_mnist` to use the MNIST data set, otherwise point to sketchy data set <br />
`--learning-rate <learning rate>` <br />
`--decay-rate <decay rate>` <br />
`--batch-size <batch size>` <br />
`--epoch-size <epoch size>` <br />
`--out <directory for storing output from generator>` <br />


### Compatability
Only tested on windows with python 3.5.2, numpy+mkl 1.12.1, scipy 0.19.0, tensorflow-gpu 1.0.1.
