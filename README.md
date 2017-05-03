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

### Running `trainer` on GCloud ML-Engine

```bash
JOB_DIR='gs://uw-cs760-dcgan/job-dir/' # directoy on GCS
JOB_ID='dcgan_job_23' # unique job-id

gcloud ml-engine jobs submit training JOB_ID \
--package-path trainer/ \
--module-name trainer.task \
--job-dir JOB_DIR \
--staging-bucket 'gs://uw-cs760-dcgan/' \
--region 'us-central1' \
--scale-tier 'BASIC_GPU' \
-- \
--data-dir '/tmp/tensorflow/mnist/input_data/'
```

### Compatability
Only tested on windows with python 3.5.2, numpy+mkl 1.12.1, scipy 0.19.0, tensorflow-gpu 1.0.1.
