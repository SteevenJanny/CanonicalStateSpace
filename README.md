# CanonicalStateSpace

This repository contains the code associated to the
paper ["Learning Reduced Nonlinear State-Space Models: an Output-Error Based
Canonical Approach", S. Janny, Q. Possamai, L. Bako, M. Nadri, C. Wolf](https://arxiv.org/abs/2206.04791).

```bibtex
@inproceeding{janny2022learning,
  title ={Learning Reduced Nonlinear State-Space Models: an Output-Error Based
Canonical Approach},
  booktitle = {The International Journal of Robotics Research},
  author = {
    Janny, Steeven and 
    Possamaï, Quentin and 
    Bako, Laurent and 
    Nadri, Madiha and 
    Wolf, Christian},
  year = {2022},
}
```

## Abstract

The identification of a nonlinear dynamic model is an open topic in control theory, especially from sparse input-output
measurements. A fundamental challenge of this problem is that very few to zero prior knowledge is available on both the
state and the nonlinear system model. To cope with this challenge, we investigate the effectiveness of deep learning in
the modeling of dynamic systems with nonlinear behavior by advocating an approach which relies on three main
ingredients: (i) we show that under some structural conditions on the to-be-identified model, the state can be expressed
in function of a sequence of the past inputs and outputs; (ii) this relation which we call the state map can be modelled
by resorting to the well-documented approximation power of deep neural networks; (iii) taking then advantage of existing
learning schemes, a state-space model can be finally identified. After the formulation and analysis of the approach, we
show its ability to identify three different nonlinear systems. The performances are evaluated in terms of open-loop
prediction on test data generated in simulation as well as a real world data-set of unmanned aerial vehicle flight
measurements.

## Training
You can train the model by running the following command:
```bash
python train_regressor.py --epoch 1000 --lr -4 --window_size 5 --n_layers 3 --name "H0"
python train_autoencoder.py --epoch 1000 --lr -4 --compression 50 --n_layers 3 --name "train_autoencoder"
```


## Dataset
Generation scripts are in the ```DataGeneration/``` folder (BlackBird raw can be downloaded separately
from [here](https://github.com/mit-aera/Blackbird-Dataset). The measurements used in the paper are available in the
```Data/``` folder.

Please cite the original paper if you use Blackbird dataset.
```bibtex
@article{antoniniIJRRblackbird,
  title ={The Blackbird UAV dataset},
  journal = {The International Journal of Robotics Research},
  author = {
    Antonini, Amado and 
    Guerra, Winter and 
    Murali, Varun and 
    Sayre-McCord, Thomas and 
    Karaman, Sertac},
  volume = {0},
  number = {0},
  pages = {0278364920908331},
  year = {0},
  doi = {10.1177/0278364920908331},
  URL = { https://doi.org/10.1177/0278364920908331 },
  eprint = { https://doi.org/10.1177/0278364920908331 }
}

@inproceedings{antonini2018blackbird,
  title={The Blackbird Dataset: A large-scale dataset for UAV perception in aggressive flight},
  booktitle={2018 International Symposium on Experimental Robotics (ISER)},
  author={
    Antonini, Amado and 
    Guerra, Winter and 
    Murali, Varun and 
    Sayre-McCord, Thomas and 
    Karaman, Sertac},
  doi={10.1007/978-3-030-33950-0_12},
  URL={ https://doi.org/10.1007/978-3-030-33950-0_12 },  
  year={2018}
}
```