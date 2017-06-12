This documents describes how to run the code.

In this project Keras library with Tensorflow Backend is used. Thus, in order to run the code successfully, first, we have to install dependencies.


1 - installing dependencies on linux or Mac OS:
	* Anaconda contains most of the library we need to run over code. Please go to the following link and install anaconda which is described based on your Operating system:
		https://www.continuum.io/downloads
	
	* install keras on Ubuntu OS or Mac OS is streightforward:
		1- Press Shift+Alt+T to open the terminal and enter the following command:
			sudo pip install keras
	* Please follow the instruction on the link below for installing dependencies on windows:
			https://goo.gl/GrCZl2

	* Please follow the instruction on the link below to install Tensorflow:

			https://www.tensorflow.org/get_started/os_setup


2- If you would like to run the code on GPU you should follow the steps on the link below to install CUDA 8.0 on your operation system:

			https://developer.nvidia.com/cuda-downloads

3- After installing all the dependencies. Please download the images of dataset from the link below, unzip the downloaded file and copy the images folder in the folder containing this REDME file.

			https://www.kaggle.com/c/leaf-classification/download/images.zip

3- For running the code please open the Terminal in Linux or Mac OS, or Anaconda Command prompt on windows OS (it can be found on start menu, installed programs):

			* To run the code enter the following command:
				python LeafClassifer.py

4- The code will start to run and it will take a while depending on how many parameters and layers you defined. After running the code it will show the best model accuracy, and loss, and save two figures one for Model accuracy and one for Model loss based on the number of iterations. The figures can be found in the same folder with this README file. Also, you can see the results and detailed graphs of architecture (on the GRAPHS tab) of the model and what is happening in each layer by entering the following command:

				 tensorboard --logdir=logs
Now you can navigate to http://127.0.1.1:6006 or http://localhost:6006 on your browser to see the results.

	
