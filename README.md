# Visualization-Tool-Mobinet
This repository contains the implementation of a visualization tool for convolutional neuronal networks.
* Initial tool can be find in https://github.com/yosinski/deep-visualization-toolbox.
* Improved version of the tool in https://github.com/arikpoz/deep-visualization-toolbox.


# Installation
This toolbox runs using Ubuntu (No special version of Ubuntu is needed).

To start with the installation open a Terminal and copy this instrucctions in the console (you migth need to introduce your github username and your password to proceed):

    $ git clone --recursive https://github.com/MBasarte/Visualization-Tool-VPU.git
 
If you dont have git installated, run:

    $ sudo apt install git

Once the repository is downloaded, next step is installing Caffe. First, you need to install some libraries if you dont have them already in your computer. We are gonna follow the instructions described in the official Caffe website (https://caffe.berkeleyvision.org/install_apt.html).

Start by cheking your Ubuntu version with the next instrucction:

    $ lsb_release -a
    
If your Ubuntu version is >= 17.04, then Caffe installation is already implemented and you only need to copy the next instruction and you can obviate the rest of the installation instructions.

    $ sudo apt install caffe-cpu

Otherwise, copy the next instructions in the console to install all the necessary libraries:

    $ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
    $ sudo apt-get install --no-install-recommends libboost-all-dev
    $ sudo apt-get install libatlas-base-dev
    $ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

Then, navigate to the foder '/caffe/python' and install the last requirements:

    $ cd Visualization-Tool-VPU/caffe/python
    $ for req in $(cat requirements.txt); do pip install $req; done
    
Once you have all the libraries installed, the compilation of Caffe is implemented in build_default.sh (file located in the folder you downloaded from this repository) but it migth be necesarry to change the file 'Makefile.config' in order to make it compatible with our system. First run the next line:

    $ cd Visualization-Tool-VPU && ./build_default.sh
   
If the instrucction its executed with no errors, Caffe is installed and compiled in your computer and you car run the visualization tool. If you get errors try the following:
(The errors solution proposed here follow the next video: https://www.youtube.com/watch?v=DnIs4DRjNL4&t=1196s. It migth be a good idea to appeal to the video for a better visualization of the solutions showed next)

   * 1-In the '/caffe' folder open the 'Makefile.config' file with the text editor you prefer (e.g. gedit)
   
   * 2-Uncomment the line 8:

            $ CPU_ONLY := 1
        
   * 3-Change the line 69 in order to make it pointing to where your numpy is installed. To locate where numpy is in your computer you can run the command:
   
            $ locate site-packages/numpy/core/include
        One alternative to find your numpy folder if the previous instruction doesn't work is to start a new python scrip by typing python in the console and writting down the next instruccions:
        
            >>> import numpy
            >>> print(numpy)
            
        It will return something similar to:
            
            <module 'numpy' from '/home/vpu/.local/lib/python2.7/site-packages/numpy/__init__.pyc'>

        Copy the direction until `/numpy/` add `core/include` and paste everything so you finish with a 69 line similar to:
  
            $ /usr/.local/lib/python2.7/site-packages/numpy/core/include
       
   * 4-Modify the line 94 to look like this:
  
            $ INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/
   
   * 5- Write this in the console
        
            $ find /usr/lib -name hdf5
        
        Copy the line anserwed by the console (it migth be something similar to `/usr/lib/aarch64-linux-gnu/hdf5` and paste it at the end of the line 95 adding the word `/serial/` at the end, so you end with a line 95 looking like:
   
            $ LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib direction_copy/serial/
   
        In my case I have:
   
            $ LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/
   
   * 6- Finally, make sure to save the 'Makefile.config' file and run again the instruction for the Caffe compilation:
   
            $ ./build_default.sh

Once caffe is compiled, you need to change the file settings_model_selector.py so the line 16 points to the direction where your caffe is installed. In my case:

    caffevis_caffe_root = ('/home/vpu/Desktop/Visualization-Tool-Mobinet/caffe')

Then, download one example model from: https://dauam-my.sharepoint.com/:u:/g/personal/marcos_escudero_uam_es/EUeTGtpC4o1Ko1xqMSFZP2cB9fdz65eLuzwBZm7hLh5hwQ?e=ZrA28K

Introduce everything downloaded in the previous step in the 'models/indoor' folder.

If everything went as planned, Caffe and the visualization tool should be installed, an you dowload one example model, so you can copy the next instruccion to run the tool:

     $ ./run_toolbox.py 

It will work with some example images located in '/input_images' folder.

Once the toolbox is running, push 'h' to show a help screen.

# Personalize

A full version of an alexnet model trained in an indoor dataset for scene recognition ready to visualize is availible at https://dauam-my.sharepoint.com/:/g/personal/marcos_escudero_uam_es/EUeTGtpC4o1Ko1xqMSFZP2cB9fdz65eLuzwBZm7hLh5hwQ?e=ZrA28K

In the next sections I will use this model as an example to adapt the tool to your own model.

## Run your model
Start by creating a folder in '/models' with the name of your network and introuce the '.prototxt' and '.caffemodel' files. In my case, I want to visualize an Alexnet model trained over an indoor dataset (http://web.mit.edu/torralba/www/indoor.html), so I create a folder in '/models' named indoor that contains the structure of my model ('alexnet_indoor.prototxt') and the value of the parameters of my model trained over the dataset ('alexnet_indoor.caffemodel').

![Screenshot](https://github.com/MBasarte/Visualization-Tool-VPU/blob/master/readme_images/alexnet1.png)


Then, create a file named 'settings_yourmodel.py' in the '/model_settings' folder. As a reference, is easier if you copy another settings file from another model and change the lines according to your own model. In my case, I created a file called 'settings_indoor.py' that looks like this:

    # basic network configuration
    base_folder = '%DVT_ROOT%/'
    caffevis_deploy_prototxt = base_folder + './models/indoor/alexnet_indoor.prototxt'
    caffevis_network_weights = base_folder + './models/indoor/alexnet_indoor.caffemodel'

    # input images
    static_files_dir = base_folder + './input_images/indoor_train'

    # UI customization
    caffevis_label_layers    = ['fc3', 'prob']
    caffevis_labels          = base_folder + './models/alexnet_indoor/categories_indoor.txt'
    caffevis_prob_layer      = 'prob'

    def caffevis_layer_pretty_name_fn(name):
        return name.replace('pool','p').replace('norm','n')

    # offline scripts configuration
    caffevis_outputs_dir = base_folder + './models/indoor/outputs'
    caffevis_outputs_dir = base_folder + './models/indoor/outputs'
    layers_to_output_in_offline_scripts = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3', 'prob']

Finally, in the file 'settings_user.py', change the `model_to_load` variable to point to your model. In my case:

    model_to_load = 'indoor'

## Load your dataset
Copy the dataset in the '/input_images' folder and change `static_files_dir` variable in the setting file of the model (in the '/model_settings' folder) to point to your dataset. Following my example I have:

    # input images
    static_files_dir = base_folder + './input_images/indoor_train'

## Generate maximal inputs for a new model
Scripts to generate images offline are located in '/find_maxes' folder. This scripts will save the desired images for the model selected in the 'settings_user.py' file.

Start by running the 'find_max_acts.py' to locate the patches that respond with higher value to the neurons of the model. 

Then run 'crop_max_patches.py' in order to save this patches (they are save in '.../models/your_model/outputs').

Finally, run 'montage_images.py' changing the model variable to your model name.

# Translate CNNs from pytorch to caffe.
If your network is developed in Pytorch, first it is necessary to translate it to Caffe. In order to do it, you can use the method proposed in https://github.com/xxradon/PytorchToCaffe.

Download the repository and select one example from the ones located in the '/example' folder-choose the one that fits better your model-. In my case, I will translate an Alexnet model trained over an indoor dataset for scene recognition (you can download my pytorch model from: https://dauam-my.sharepoint.com/:f:/g/personal/marcos_escudero_uam_es/EsRjhfBVaKlKnobdtG2YhiwBRE587ZmRDd2fn_TZ-lXDJA?e=UvftYSThen). I select the example 'alexnet_pytorch_to_caffe.py' and I change the parameters according to my model:

    net=('/home/vpu/Downloads/alexnet_indoor.pth')
    input=Variable(torch.ones([1,3,254,254]))

Then I run the example script. If you dont have pytorch installed, follow the instructions described in https://pytorch.org/get-started/locally/-select the pytorch version according to your computer requisites-.
 
Once the translation has finished, you have a .caffemodel file-define the weights and bias of the model- and a .protxt file-define the structure of the network-. These files need to be relocated in the '/models' folder, following the Personalize instructions in this readme if you want to visualize the model.

If you get an error similar to this one:

    F0607 13:28:49.344043 5379 reshape_layer.cpp:87] Check failed: top[0]->count() == bottom[0]->count() (9216 vs. 1024) output count must match input count
    *** Check failure stack trace: ***
    Aborted (core dumped)

Its due to a bad translation of the structure of the model. To solve this problem, We can take advantage of the fact that the structure of a model is independent of the parameters and does not vary in training so we can use an existing structure of our model. In my case, I downloaded the 'Alexnet.prototxt' from a network trained for another task and I used as my model structure.
There are many examples on the web of most commonly used structures.

 
