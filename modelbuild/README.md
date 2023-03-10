## Overview of this Project and its components

This project provides a starting point for creating a CICD system for machine learning. There are three primary components to this project:

* Image generation
* Model building
* Model deployment

Each of the above components is made up of:

* A CodeCommit repository (like this one)
* A CodeBuild project
* A CodePipeline
* Various SageMaker components

This repository is for the project's **Model building** component. It was created as part of a SageMaker Project.

In the placeholder code provided in this repository, we are solving the abalone age prediction problem using the abalone dataset.
## Layout of the model building repository 

```
|-- codebuild-buildspec.yml
|-- CONTRIBUTING.md
|-- pipelines
|   |-- abalone
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- preprocess.py
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   |-- __version__.py
|-- README.md
|-- sagemaker-pipelines-project.ipynb
|-- setup.cfg
|-- setup.py
|-- tests
|   |-- test_pipelines.py
|-- tox.ini
```

This section provides an overview of how the code is organized and the purpose of important files. In particular, `pipelines/pipelines.py` contains the core of the business logic for this problem. It has the code to express the ML steps involved in generating an ML model. You will also find the code that supports preprocessing and evaluation steps in `preprocess.py` and `evaluate.py` files respectively.

Once you understand the code structure described below, you can inspect the code and can start customizing it for your own business case. This is only sample code--you own this repository and can adapt it for your business use case. Feel free to modify the files, commit your changes and watch as the CICD system reacts to them.

You can also use the `sagemaker-pipelines-project.ipynb` notebook to experiment from SageMaker Studio before you are ready to checkin your code.

```
|-- codebuild-buildspec.yml
```
Your codebuild execution instructions. This file contains the instructions needed to create or update a SageMaker Model Building pipeline and start an execution of it. You will see that this file has the fields definined for naming the Pipeline, ModelPackageGroup etc. You can customize them as required.

```
|-- pipelines
|   |-- abalone
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- preprocess.py
```
Your pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `get_pipeline` interface as illustrated here.

```
|-- pipelines
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   |-- __version__.py
```
Utility modules for getting the pipeline definition JSON and running pipelines (you do not typically need to modify these):

```
|-- setup.cfg
|-- setup.py
```
Standard Python package artifacts.

```
|-- tests
|   |-- test_pipelines.py
```
A stubbed testing module for testing your pipeline as you develop.

```
`-- tox.ini
```
The `tox` testing framework configuration.
## Describe dataset used for pipeline below.
# TODO

