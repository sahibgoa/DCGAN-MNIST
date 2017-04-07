# send-doods

### About
Generative Adversarial Network for doodle creation. Discriminator discriminates sketched photos from photographs. Network for doodle generating coming soon. Images of sketches and photos downloaded from http://sketchy.eye.gatech.edu/. First 25 images of each category kept for reasonable RAM usage and repository size.

### Dependencies
python, numpy, scipy, tensorflow (gpu version recommended), pillow

### Running
For the discriminator, call 'python discriminator.py' from the command line to train the network and see its accuracy evaluations.

##### Compatability
Only tested on windows with python 3.5.2, numpy 1.12.1, scipy 0.19.0, tensorflow-gpu 1.0.1, pillow 4.1.0.