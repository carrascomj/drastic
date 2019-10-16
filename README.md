# drastic
Deep Rapid Annotation using Submodels Training in Cells

What is drastic?
---------------

Drastic provides a command line tool to perform annotation of a prokaryotic genome
based on Deep Learning. This project is highly inspired DeepAnnotator [[1]](#amin2019) (repository [here](https://github.com/ruhulsbu/DeepAnnotator)).

Getting started
---------------
How to clone, add `bin` to the `PATH` and use the model to train a model here.

Capabilities
------------
This project aims to provide a deep neural network capable of annotating a prokaryotic
genome.
* Trained LSTM neural network model to annotate user provided genome.
* Score performance based on [[1]](#amin2019).

Alternatively, the user could provide its own data to train the model.
* Pre-process data supplied by the user.
* Train the model with this data.
* Use the model to predict.

Command line options
--------------------
Provided as help, in this section I should reference

Schema of the model
-------------------
An explanation (better with a graph) should be placed here.

What I Learned
-------------
This project was developed for the MSc course "Deep Learning" of the Technical University of Denmark.
In this section, I will summarise the things I have learnt.

References
---------

<a name="amin2019">1</a>: Amin, M. R., Yurovsky, A., Tian, Y., & Skiena, S. (2018). DeepAnnotator. The 2018 ACM International Conference
