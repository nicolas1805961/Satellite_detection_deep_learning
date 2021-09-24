To install detectron:

First install Pytorch:

    - pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

Then Detectron2 can be installed with the following command line:

    - python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

The config file is divided in 3 separate parts:

    - Training
    - Inference
    - Benchmarks/Graphs

Some parameters must be tuned in each part of this config file. Benchmarks and graphs are not available for instance segmentation.


## train_sst.py

usage: python train_sst.py config_file

This file is used to train the model

config_file is a mandatory argument specifying the config file path.

The PATH_TRAIN argument in the config file specifies the path to the folder containing training data. This folder must contain 3 subfolders:

    - 'png_data', containing training images. Images should be named i.png where i belongs to [0; number of images].
    - 'csv_data', which contains csv files matching each image. csv files should be named i.png where i belongs to [0; number of images].
    - 'json_data', which contains json files matching each image. json files should be named i.png where i belongs to [0; number of images].

As part of instance segmentation, a fourth subfolder should be present:

    - 'label' containing labeled/segmented images from the simulator. Images should be named i.png where i belongs to [0; number of images].

The 'i's of these 3 or 4 subfolders should match. For instance 0.png (image), 0.csv (stars), 0.json (satellites) and 0.png (labeled images) is data matching the same image.

To launch training on only one or two classes, the line 70 and 113 of the file 'train_sst.py' must be modified for bounding box detection and instance segmentation respectively. The modification should be the following:

    - for cfg_class in cfgs[] --> specify within the brackets which class to iterate on. cfgs is a list containing 3 or 2 detectors depending on the acquisition mode.

The 'inf/nan training has diverged' error means that the learning rate is too high, it should be decreased.


## inference_real_image.py

usage: python inference_real_image.py config_file

This file make it possible to infer on real images

config_file is a mandatory argument specifying the config file path.

The PATH_TEST argument in the config file specifies the path to the folder containing images used for inference. Each and every image in this folder will be read. Images need not be named in any specific way.

The WEIGHT_FOLDER argument in the config file specifies the path to the folder containing the model's weights. This folder must include:

    - 'cfgs.txt'. A binary file that contains training and testing parameters

    - 'output_defilant'. A folder which includes the 'model_final.pth' file. This file contains the model's weights for the detection of  trailing satellites.

    - 'output_ponctuel'. A folder which includes the 'model_final.pth' file. This file contains the model's weights for the detection of point-like satellites.

    - 'output_star'. A folder which includes the 'model_final.pth' file. This file contains the model's weights for star detection.

By default, the model's weights are located in the 'good_weights' folder at /data_deep/SST_CNES_LOT2/


## run_benchmark.py:

usage: python run_benchmark.py config_file

This file allows to launch benchmarks on testing data.

config_file is a mandatory argument specifying the config file path.

The PATH_BENCH argument in the config file specifies the path to the testing folder. This folder will be used to run deep learning algorithm's benchmarks. This folder's structure must be the same as PATH_TRAIN (see above).

The PATH_TRITON argument in the config file specifies the path to the folder containing the files with results of the triton algorithm. These files must be in a json format. They must be named i.json with i in [0; number of images]. The 'i's in PATH_TRITON must match with the 'i's of PATH_BENCH. For example, 0.json in PATH_TRITON must match with 0.json, 0.csv and 0.png in PATH_BENCH. The file numbers in these two paths must match with the corresponding testing images (so images in the PATH_BENCH folder).

The WEIGHT_FOLDER argument in the config file specifies the path to the folder containing the model's weights. This folder must include:

    - 'cfgs.txt'. A binary file that contains training and testing parameters

    - 'output_defilant'. A folder which includes the 'model_final.pth' file. This file contains the model's weights for the detection of  trailing satellites.

    - 'output_ponctuel'. A folder which includes the 'model_final.pth' file. This file contains the model's weights for the detection of point-like satellites.

    - 'output_star'. A folder which includes the 'model_final.pth' file. This file contains the model's weights for star detection.

METHOD_1_NAME and METHOD_2_NAME in the config file are the compared methods name. If one of the methods used for comparison is triton, this field must take the 'TRITON' value.

The WEIGHT_FOLDER_1 field of the config file is the path to the model's weights whose name is METHOD_1_NAME. If METHOD_1_NAME takes the value 'TRITON', WEIGHT_FOLDER_1 is ignored.

The WEIGHT_FOLDER_2 field of the config file is the path to the model's weights whose name is METHOD_2_NAME. If METHOD_2_NAME takes the value 'TRITON', WEIGHT_FOLDER_2 is ignored.

## graph.py

usage: python graph.py config_file

This file creates the comparison graph for both methods as well as quantitative results in separate files for each algorithm.

config_file is a mandatory argument specifying the config file path.

Statistics used to create graphs and result files are computed by the 'run_benchmark.py' file. The 'graph.py' file reads the statistics located at PATH_OUT and saves graphs at the same location.

This file, which makes it possible to create graphs, must be launched **AFTER** having executed the 'run_benchmark.py' file which saves results.

The config file's fields used to create graphs are those in the benchmark part. That is:

    - The MODE field which must take the same value when generating graphs and running benchmarks ('run_benchmark.py' executed before).

    - The METHOD_1_NAME and METHOD_2_NAME fields which must also take the same value for graph creation and benchmarking (see 'run_benchmark.py' part)

    - The PATH_OUT field that specifies the path where graphs will be saved.


There are two pairs of file to launch training on Polyaxon.

The 'polyaxon_create_image_scripts.yaml' and 'polyaxon_run_scripts.yaml' allow to launch training on Polyaxon based on 'py' files.

The 'polyaxon_create_image_notebooks.yaml' and 'polyaxon_run_notebooks.yaml' allow to use notebooks in Polyaxon.

In both cases, the 'polyaxon_create_image_*.yaml' must be executed **BEFORE** the 'polyaxon_run_*.yaml' file.

When 'py' files are executed on Polyaxon, the paths within the config file ('config.yaml') must start with '/data_deep/' instead of '/run/'.

If 'py' files are to be executed on Polyaxon, source code and config file must be located in a folder on '/data_deep'. Then, in 'polyaxon_run_script.yaml', the path to that directory must be modified ('context_folder' field).