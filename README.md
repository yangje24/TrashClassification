This is the repository for our EECS 498 Mobile Trash can for Waste Classification project.
The training branch contains the code and some of the settings used to train the YOLO 26 models.
The main branch contains the code used to run the inference and the Raspberry Pi side code.

computer_side_small_enhanced.py is the code used during inference and stream_server_small.py is the Raspberry Pi stream server code

distance_calculation.py contains the code originally used when we planned to have the robot follow a thrown object and needed to know how far away the object is

webcam_inference.py is used for testing the YOLO models through a webcam

The different .pt files are trained weights for different YOLO models and the names are based on the parameters they were trained with in terms of epochs and size

The remove_labels and remove_images.py in the training branch were used to try and remove some less useful annotations from the dataset

convert_model.py was used to convert models to an ncnn model which is lighter weight and only uses the CPU which was useful when we wanted to run the models on the Raspberry Pi
