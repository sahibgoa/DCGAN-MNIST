# send-doods

### About
Generative Adversarial Network implementation.

### Dependencies
python, numpy, scipy, tensorflow (gpu version recommended).

### Running
`python main.py`

arguments
`--data_dir <directory for storing input data>`
`--use_mnist` to use the MNIST data set, otherwise point to sketchy data set
`--learning-rate <learning rate>`
`--decay-rate <decay rate>`
`--batch-size <batch size>`
`--epoch-size <epoch size>`
`--out <directory for storing output from generator>`


### Compatability
Only tested on windows with python 3.5.2, numpy+mkl 1.12.1, scipy 0.19.0, tensorflow-gpu 1.0.1.
