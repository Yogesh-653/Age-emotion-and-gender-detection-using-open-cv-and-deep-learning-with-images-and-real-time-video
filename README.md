                                          Age, Emotion and Gender detection using Deep learning and Open CV
Abstract

The human beings has the advance visual observation of this century to start unimaginable discovers. The process of visual observation using the camera, images and video it is possible to determine the age, gender and emotion of the individual to utilize for the further many more application. This project will go under the entire process including the multiple method and deep learning techniques to check the accuracy level and how it all come together. Then it will highlight the important and how its implemented to our day today life. 

The main objective of this paper is to build to age, emotion and gender detector that can approximately detect the age, emotion and gender of the face of an individual in the picture as in the dataset using the deep learning models. The application of the model mostly implemented in the security services cum CCTV surveillance. Importantly for the emotion detection will mainly uses in the hospitality for analysing the patient emotion expression under intensive care unit(ICU) .This project introduces the sophisticated system that utilises the Convolutional Neural Network (CNN) model to explore various set of methods follows by the study presents a novel system that combines with You Only Look Once (YOLO) model for detecting the region of the facial structure  and Wide Residual Network (WideResNet) and used to detect the age and gender. Particularly for emotion detection we are using emotion classifier pre-trained model. For building the model we created the featured dataset that will match under the test images and will detect the from the test sample images. By the way we are using the three states of input for find the age, emotion and gender detection 1. Importing input images from the dataset of the audience that was taken from the UTKFace 2.Importing pre-recorded video 3.Real time live video recording   
1	Introduction
The fast growth of Artificial Intelligence (AI) and Computer Vision (CV) and their widespread use in many areas has had a hugely positive effect on many areas of human life. In real time world videos and images are the input feed for the everyday task going under the surveillance for the security and monitoring the facial expression how people feeling their emotions. In the past few years, a lot of study has been done on using facial expressions to find out certain things about faces. In order to improve classification accuracy and loss of regression, it is customary to extract facial features, a task that is both delicate and challenging. However, our methodology emphasised head detection rather than solely the frontal face, enabling us to extract features from the hair, ear, and head surface in addition to faces. Technology that Deep learning model and Open CV used to objective is to discern facial expressions, gender, and age from identical images and video.

	The facial recognition from the human face is part of input to detecting age, emotion and gender. Hardware that improves the advance obtained for this project applying Convolutional Neural Networks (CNN) model for detecting inputs with more layer popularised in deep learning. Similarly, the advancements in CNN architectures have been pivotal in emotion recognition, enabling the models to learn and identify various emotional states from facial expressions with remarkable accuracy (Lopes et al., 2017). Furthermore, the inclusion of the YOLO (You Only Look Once) model, renowned for its real-time object detection capabilities, presents an innovative approach to identifying faces from different inputs, including still images and video streams. The YOLO model's integration allows for rapid and efficient processing, making it suitable for real-time applications (Redmon et al., 2016). Wide Residual Networks (WideResNet), known for its depth and widened layers, offers a significant advantage in capturing complex patterns in facial data, which is crucial for accurate age and gender prediction (Zhang et al., 2016). Then using emotion classifier model to detect the emotion of the humane face. From this perspective the work analyses how a state-of-the-art age, emotion and gender detection with the low quality of facial input images and videos for the application used for the surveillance of security and monitoring emotion. Additionally, this project improvises the model with various methods including speed, accuracy and loss of error to find the best match. 
1.1 Problem Statement
     This project aims to answer the following research question “How a state-of-the-art Age, emotion and gender detection done with deep learning algorithm and how computer vision works when limited and low-quality input images and video are available?” 
1.2 Project Objectives
This research objective is purely based on the working of deep learning model and open computer vision. To detect the age, emotion and gender with human facial data and expression.

•	Create a human facial dataset to detect age, emotion and gender
•	Analyse state of art techniques of face detection by chosen dataset
•	Based on state-of-the-art techniques, develop a model age and emotion
•	Based on state-of-the-art techniques implementing pre-trained of emotion classifier to detect the emotion from the dataset 
•	Define strategy for training, validation and testing the model


•	Testing those models with sample input images to 
•	Examine the performance of the model
•	working of real time live video as input to detect age, emotion and gender by using the model we trained
1.3 Overview of This Report

This project is divided into ten chapters, Chapter 2 presents the literature review, It extract current state-of-the-art age, emotion and gender detection methods and model used. Chapter 3 has the methodology were the various methods goes on to demonstrate this project including the requirements, dataset and training, testing of the model. Then types of input sources and various result and discussion was done these are the major part worked in this methodology. Chapter 8 has Critical Appraisal, chapter 9 with Conclusion including achievements and a reflection on possible future work to this project. The final comes student reflection and bibliography with comes under 10 chapter.



 
2	Literature Review

2.1 Computer Vision (CV)

Computer vision is the field around the artificial intelligence that helps the computer and enables to have a vision on images and video then identify everything as like human would. This machine interprets and make a decision based on the visual the data which replicate and exceed the capabilities of human vision using computer. Computer vision majorly involves acquiring, processing, analysing and understanding digital images to extract data from the real world in order to generate symbolic or numerical information that it uses to make decisions on. This process includes different practices like object recognition, tracking a video, motion estimation and image restoration. The evolution of computer vision spans several decades, starting from the 1960s with basic image processing techniques. The initial focus was on pattern recognition, which laid the foundation for more complex applications. By the 1980s and 1990s, the field had progressed to include feature extraction and object recognition. The seminal work by Canny in 1986 on edge detection is a notable milestone, widely used even today (Canny, 1986). Thus, the given Figure 1. Is the computer vision of face detection.

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/51d69aa1-dc20-4ef7-814e-944ae5a5cc1b)

A significant breakthrough in computer vision came with the advent of deep learning, particularly with the development of Convolutional Neural Networks (CNNs). (Krizhevsky et al. 2012) demonstrated the power of CNNs in image classification, significantly outperforming previous methods in the ImageNet challenge. This marked a paradigm shift in the field, with deep learning becoming the core methodology in computer vision tasks. Advanced age and gender detection models have evolved from simple binary classifiers to sophisticated systems capable of accurately predicting a wide range of ages and recognizing gender with high precision. (Rothe et al. 2018) introduced the Deep EXpectation (DEX) algorithm, which utilizes a deep CNN trained on a large dataset to estimate age. Their model achieved state-of-the-art performance on standard benchmark datasets. Similarly, for gender detection, the work by (Han et al. 2017) demonstrated the use of CNNs to achieve remarkable accuracy, even under challenging conditions such as varying lighting and occlusions.
Emotion detection has progressed from basic facial expression analysis to nuanced understanding of subtle emotional cues. In a landmark study, (Li and Deng et al.2019) utilized a hybrid CNN-RNN architecture, which not only analysed static facial features but also captured the temporal dynamics of facial expressions, leading to a more accurate interpretation of emotions. This approach exemplifies the trend towards integrating spatial and temporal data for richer emotional analysis. These features are square-shaped functions that define rectangles inside an image window they resemble Haar wavelets. The feature value is calculated as the difference between the sum of the pixel values in the white and black sections of the rectangles. Because the nose bridge region is usually brighter than the eyes and the eye region is usually darker than the cheeks, this method works especially well for face detection.

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/abd57e11-5604-4b15-b032-dec4eca845b1)


Figure 2. Is the structure layers of the computer vision working control from raw input to the user gets the output of the result. The integration of deep learning into computer vision has led to significant advancements in object detection, image localization, semantic segmentation, and pose estimation. Object detection has evolved into more efficient forms, such as one-step detection (e.g., YOLO (Redmon et al., 2016), SSD (Liu et al., 2016), RetinaNet (Lin et al., 2017)) that streamline the process by combining detection and classification. In image localization and object detection, CNN architectures like AlexNet (Krizhevsky et al., 2012) and Faster RCNN (Ren et al., 2015) are pivotal, enabling precise object identification and categorization, essential in fields like medical diagnostics. Semantic segmentation further refines this by identifying objects at the pixel level without relying on bounding boxes, using methods like FCN (Long et al., 2015) and U-Nets (Ronneberger et al., 2015), crucial in autonomous vehicle training. Lastly, pose estimation, powered by architectures like PoseNet (Papandreou et al., 2017), identifies joint positions in images, facilitating applications in augmented reality and gait analysis. These advancements collectively demonstrate deep learning's transformative impact on computer vision, enhancing accuracy and expanding application possibilities.

2.2 Convolutional Neural Network (CNN):
	
	Convolutional Neural Networks (CNNs), a class of deep neural networks, have significantly advanced the field of image processing. Central to tasks such as image classification, segmentation, and object detection, these networks represent a cornerstone in modern computer vision [Sikder et al., 2021]. The architecture of CNNs is distinctively designed to mimic the human visual cortex, enabling effective feature extraction and pattern recognition in visual data.  At the core of a CNN's architecture are layers that each play a specific role in the processing and interpretation of images. The fundamental layers include the input layer, convolutional layers, pooling layers, fully connected layers, and the output layer. The pattern of these layers typically follows the structure (1)

InL=>[ConvL=>PoL?]*X=>[FCL]*Y=>Out L(1)                (1)
		
In this equation (1), InL represents the Input Layer, ConvL the Convolutional Layer, PoL the Pooling Layer, FCL the Fully Connected Layer, and OutL the Output Layer. The variables X and Y denote the number of times these layers are repeated in the architecture, offering flexibility and adaptability in the network's design [Antipov et al., 2017]. The figure.3 represents the architecture of CNN


![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/3ec58ef8-3017-46da-b4e6-2faae1e1fcd5)
Figure 3 The proposed CNN Model Architecture.
The convolutional layers are where the network performs most of its heavy lifting. Here, filters, or kernels, move across the input image, analysing small portions at a time. Each filter detects features at different spatial hierarchies, from basic edges in the initial layers to complex patterns in the deeper layers. The output of these convolutional operations is a set of feature maps that represent the presence of specific features in the input image [LeCun et al., 1998]. The following convolutional layers are the pooling layers reduce the spatial dimensions (width and height) of the input volume for the next convolutional layer. The two common types of pooling are Max Pooling and Average Pooling, where the former is more commonly used due to its effectiveness in feature representation. In figure.4 shows Pooling helps in reducing computational complexity and in making the detection of features invariant to scale and orientation changes [Scherer et al., 2010]. 
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/3055be17-8451-446d-94fa-8f88d4b3a88f)

The fully connected layer (FC) is responsible for performing the predictions. FC layer
neurons are connected to all the activations of the previous layer. For a classification problem, this layer is responsible for computing the score of each class and outputting the winner. After several convolutional and pooling layers, the network uses fully connected layers. Here, neurons have full connections to all activations in the previous layer, as seen in regular Neural Networks. These layers are typically placed towards the end of the CNN architecture and are used to perform high-level reasoning based on the features extracted by the convolutional and pooling layers. The final fully connected layer holds the output, such as the classification scores [Krizhevsky et al., 2012].

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/fa0b4f31-7d74-416d-9796-09942e54d2fd)
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/77638216-ad80-49cd-b16f-415876b001bb)

2.3 Wide Residual Network (Wide ResNet)

	The convolution neural network major in deep learning. The architecture design for the image recognition like ResNet [He et al., 2016] have been successfully used in the computer vision. The wide residual neural network [Zagoruyko and Komodakis, 2016] used to tell the difference between fall and optical flow. The foremost portion in the neural network is deeper called the wider residual units provides much more effective way to improve performance of residual network compared to increase their layer depth. The wide residual unit mapping expressed has in below equation(2)

zl+1=zl+F(zl,ωl)                        (2)

where equation (2) zl+1 and zl are input and output of the l-th unit in the network, F is a residual function and col are parameters of the block. The wide residual network is sequentially made up of stacked wide residual blocks which include two consecutive 3 × 3 convolutions with batch normalization and ReLU preceding convolution (as shown in Figure. 6b). Compared with the original architecture in residual network (as shown in Figure. 6a), the order of batch normalization, activation and convolution is different that conv-BN-ReLU is changed to BN-ReLU-conv. The latter can be trained faster and meanwhile can achieve better performance, which is beneficial for detecting fall event in real time.

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/32538fe1-f230-42ba-a281-21a79d27cebf)

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/bba2646e-fd64-4197-a687-85e83ab8aee6)

Wide ResNet architecture with a depth of 16 and a width of 2. The notation (k×k,n) in the convolutional block and residual blocks denotes a ﬁlter of size k and n channels. The dimensionality of outputs from each block is also annotated. The detailed structure of the residual block is shown in the dashed line box. Note that batch normalization and ReLU precede the convolution layers and fully connected layer but omitted in the ﬁgure for clarity. The main architecture of the Wide ResNet is in figure.7 

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/b258363b-6926-4b4a-a0d3-491cb95da512)


Figure 7 Wide ResNet architecture
Wide ResNets have shown remarkable performance in various image processing tasks, especially in image classification challenges. They have been particularly effective in reducing the error rates on benchmark datasets like CIFAR-10 and CIFAR-100, outperforming deeper networks. The efficiency and accuracy of Wide ResNets make them suitable for applications where both speed and high performance are crucial, such as real-time image processing and face detection. 
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/b704e7ee-589f-4a0e-b1fe-90bd90d360e6)

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/44ec7eda-bfe0-4c6a-ac5b-82d7a383f9f9)

The Figure 8 in discussion presents a comparative analysis of various pruning methods applied to two well-established CNN architectures: ResNet-56 and VGG-16, benchmarked on CIFAR-10 and CIFAR-100 datasets. CIFAR-10 consists of 60,000 32x32 colour images in 10 different classes, while CIFAR-100 is similar but with 100 classes. These datasets are widely used in machine learning for evaluating the performance of image classification algorithms due to their moderate complexity and size, which makes them suitable for rapid experimentation [He et al., 2016]. The results for Wide ResNets on CIFAR-10 and CIFAR-100 demonstrate that some pruning methods can reduce complexity by up to 51% for CIFAR-10 and 25% for CIFAR-100, while still achieving the best accuracy [Liu et al., 2017]. In some instances, the pruned models even slightly outperformed the baseline [Zagoruyko and Komodakis, 2016]. This indicates that pruning can sometimes enhance network efficiency and effectiveness by eliminating redundant or non-informative parameters. These findings underscore the potential of Wide ResNet in applications where model efficiency is paramount, such as in mobile and embedded vision systems, where computational resources are limited. Furthermore, the enhanced performance of pruned Wide ResNets in image classification tasks suggests their utility in real-time image processing and object detection, such as in autonomous vehicle systems where rapid and accurate interpretation of visual information is essential for safe operation [Wang et al., 2019].

2.4 You Only Look Once (YOLO)

	In 2016, it was presented a novel approach to object detection named You Only Look Once (Redmon et al. 2016). YOLO has been through several iterations, with YOLOv5 being one of the latest. YOLOv5 has made significant improvements over its predecessors in terms of speed and accuracy, which are critical for applications requiring real-time analysis [Ultralytics, 2020]. YOLOv5 builds upon the improvements made in previous versions, optimizing the network architecture for both speed and accuracy. It includes changes to the backbone, neck, and head of the network, resulting in a more efficient model [Bochkovskiy et al., 2020]. The use of Cross Stage Partial networks (CSP) in the backbone reduces the computational cost while maintaining accuracy. Additionally, the incorporation of spatial pyramid pooling and path aggregation network techniques in the neck improves the model's ability to detect objects at various scales [Wang et al., 2020]. YOLOv5's architecture is designed to be scalable, allowing for different model sizes that provide a trade-off between speed and accuracy, catering to the requirements of various hardware platforms [Ultralytics, 2020].

The Figure.9 YOLOv5 architecture, known for its real-time object detection capabilities, is structured into three primary sections: the backbone, the neck (which includes PANet), and the output layers, each with a distinctive role in processing the input image [Redmon et al., 2016]. The backbone serves as the feature extractor, utilizing Bottleneck CSP blocks for efficient computation and the Spatial Pyramid Pooling (SPP) layer to capture multi-scale spatial features, which is vital for detecting objects of various sizes [Bochkovskiy et al., 2020].
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/d9e8cf4b-42db-47fd-b7d8-9719561213d7)

Figure 9 Overview of YOLOv5

The PANet section enhances feature propagation and reuse, employing up sample and Concatenation operations alongside convolutional layers to merge semantic information from different levels of the network [Liu et al., 2018]. The output layers use 1x1 convolutions to refine the feature map's depth, preparing it for the final bounding box and class probability predictions. This systematic approach allows YOLOv5 to deliver precise detections across different object scales rapidly, making it highly effective for applications that demand both speed and accuracy, such as autonomous vehicles and surveillance systems [Jocher et al., 2020].

	The input image is partitioned into a uniform grid of S∗S cells, where each cell is defined by its coordinates (x, y), width (w), height (h), and confidence score (C) indicating the presence of an object. The coordinates (x,y) indicate the location of the centre of the detection border box in relation to the grid. The variables (w, h) represent the dimensions of the detection border box, specifically its width and height. Every grid provides a forecast of the likelihood of C categories. The confidence score indicates the likelihood of the model including the target object and the precision of the forecast detection box. The notation Pr(item) represents the probability of a target item being present in a given cell. Confidence is described as a state of assurance or certainty.
The formula (3) represents the conditional probability of an object given the intersection over union of the predicted and true objects.

			C(Object)=Pr(Object)IOU(Pred,Truth)              (3)

If it is established that the cell does not contain a target object, the confidence score should be set to zero, denoted as C(Object) = 0. IOU, is a metric that measures the degree of overlap between a generated candidate bound and a ground truth bound. It is calculated by taking the ratio of their intersection to their union.

IOU(Pred,Truth)=area(boxmth)∩area(boxpred)/area(boxWuth)∪area(boxpred)   (4)


The prediction boxes are evaluated to determine their confidence levels. Boxes with low scores are eliminated by applying a threshold value. The remaining bounding boxes undergo non-maximum suppression. The YOLOv5 model is available in four primary variants: small (s), medium (m), large (l), and extra-large (x), with each version exhibiting increasingly improved accuracy levels. Each type also requires a distinct duration for training in the dataset of COCO
The objective of the chart is to create an object detector model that achieves high performance (Y-axis) in relation to its inference time (X-axis). Initial findings indicate that YOLOv5 outperforms other cutting-edge approaches in achieving this objective.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/3fc15e4e-62dc-462d-932f-92418a306f1c)

As figure10 the YOLOv5s used for this project to done with preprocessing step. The chart above clearly demonstrates that all iterations of YOLOv5 exhibit superior training speed compared to EfficientDet. The YOLOv5x model, known for its high precision, can process images at a significantly faster rate compared to the EfficientDet D4 model, while maintaining a similar level of accuracy. This data will be further elaborated upon later in the essay. The performance enhancement of YOLOv5 primarily stems from the PyTorch training techniques, whereas the model architecture closely resembles that of YOLOv4.

The graph presents a performance evaluation of the YOLOv5 model variants against EfficientDet, indicating their accuracy and speed on the COCO benchmark dataset for object detection [Tan et al., 2020; Jocher et al., 2020]. The vertical axis measures the COCO Average Precision (AP), a key indicator of model accuracy, while the horizontal axis displays GPU inference speed in milliseconds per image, a critical factor for real-time application feasibility. The YOLOv5 family, including the small (YOLOv5s), medium (YOLOv5m), large (YOLOv5l), and extra-large (YOLOv5x) configurations, shows a spectrum of performance trade-offs. Notably, the YOLOv5l variant demonstrates a balance between high precision and rapid processing, marking it as a suitable choice for scenarios demanding prompt and reliable object detection. In comparison, EfficientDet models, depicted from D0 through D4, offer incremental enhancements in efficiency [Tan et al., 2020]. The comparative analysis highlights the YOLOv5l as particularly effective, rivalling the efficiency of EfficientDet while maintaining commendable accuracy, which underscores its potential in real-time processing tasks such as those required in autonomous vehicle navigation and surveillance systems [Jocher et al., 2020].


2.5 Conclusion

Computer vision has been substantially transformed by the advancements in neural network architectures such as CNNs, Wide ResNets, and YOLOv5. CNNs have established themselves as the backbone of image recognition tasks, bringing forth an era of highly accurate and efficient image analysis through innovations in layer design and network depth. Wide ResNets further refined these capabilities by introducing a trade-off between depth and width, proving that wider, shallower networks can achieve, and sometimes surpass, the performance of their deeper counterparts, particularly in terms of speed and computational efficiency. YOLOv5 represents the culmination of these developments in the field of real-time object detection, showcasing remarkable precision and speed, even on standard benchmark datasets like COCO

3	Methodology

This methodology chapter applied in this project. The best methodology is defined under the modelling process to this project. The remaining of this chapter describers the dataset creation, architecture of the detections, Model, testing and training strategy, hardware and code implementation.

3.1 Dataset Creation
	For this project based on facial recognition three set of input given has the data to differentiate various output of the detection to get used in their concern application.

3.1.1 Image as input from the dataset: 

	The dataset used in this project is UTKFace dataset which it has the large-scale face dataset is found with high age span from 0- to 116-year-old. The dataset which it contains of more than 20,000 face images with mention label age, gender and ethnicity. The images cover large variation of posture, face expression, lighting, occlusion, and resolution. This dataset used for this project to age estimation, face detection, gender detection and emotion detection. Where age estimation is mentioned has the regression method and gender is done by the classification method. It provides the corresponding aligned and cropped faces with images label by age and gender

Label:
The labels of each face image is attached in the file name, formatted like
[age]_[gender]_[race]_[date&time].jpg

•	Age is an integer between 0 and 116, denoting the age; 
•	gender is an integer between 0 for male and 1 for female, denoting the gender; 
•	race is an integer between 0 and 4, denoting the different country regions' populations (0 to 4, denoting White, Black, Asian, and Indian).
•	date&time is displayed on UTKFace as (yyyy)(mm)(dd)(HHMMSSFFF), indicating the day and time an image was taken.

For this project Age and gender label gets extracted and used for the training and testing to creating model remaining label is get extracted from original label. The figure 11. Denotes the single images of the dataset as a sample where 24 is the age and next 0 which it was gender then remaining its people region which it was 2 and 20170116165047009 which was time and date

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/1b8fd1f3-062b-4673-9a71-e23d1584fec0)

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/2c4d4383-8974-4e6c-abc2-3e946227f001)
Based on the Exploratory Data Analysis (EDA) conducted on the dataset, we've gained valuable insights into the age and gender distribution of the images. In Figure 12.  age distribution is visualized through a bar chart, which reveals a prominent concentration of ages around the early twenties, with a gradual decline as age increases. In figure 13 box plot provides a clear summary of the age distribution, highlighting the median age, the interquartile range, and potential outliers.

In figure 14. gender distribution is roughly balanced, with a pie chart indicating a slight male majority at 54% compared to 46% female. These visualizations are crucial for understanding the demographic makeup of the dataset, which is essential for tasks such as developing age and gender recognition models.

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/7bba0d58-00c7-44ac-a4eb-8550dd2f2192)
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/72309e7c-3a65-4e7b-9af8-5353df6bbdbd)

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/2f2e93d7-505e-46a2-8b57-328206e83fe2)


3.1.2 Input as the video file:
		This input video data based on the indication of human being who performing news reading around one-minute flash news as format of MP4 file and frame rate around 25.00 per second where this video file attached under the dataset drive. This data took from the video stream file from the open source you tube which video is under the high-quality resource which perfectly matching under the all source of the lighting and resolution  

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/cc27f0f6-9d94-4aa1-997d-e7878ed48cbf)

Figure 14 Sample Input video data
In figure 14 displays the video snap of the news reader detector function is used to detect faces in each frame of the video. This detector locates faces and provides a bounding box around detected face. Open CV function opens the video file and reads the frame one by one loop. The detected face coordinates are expanded using margin to include some context around the face which can help in detection. The face region extracted and resized to 64x64. For each detected face the region is converted to grayscale since emotion detection model expects a single channel input. The grayscale image is resized to 48 x 48 which the input size expected by the emotion detection model. The image is normalized pixel value scaled between 0 and 1, converted to an array and expanded to have an additional dimension to represent batch size





3.1.3 Input as real time video:

Another set of input which this time it was real time video where it goes under live format for this real time, we can consider two type video one is CCTV surveillance Figure 16 and another is real time video that usable under webcam live from the face recognition opposite human who is interacting through webcam as displayed in figure15. In this project for this real time video work under process of input feed from the webcam. Open Computer Vision function used to capture video from the webcam within loop frames are continuously read from web camera then convert it to the grayscale. Each detected face computes an extended bounding box by applying a margin to include some context around the detected face then crop the extended face region from the frame resize cropped face image to 64x64 input size for the age, gender and emotion detection and applying model to predict. This application effectively functions as an interactive system that processes live video data to detect and annotate human faces with age, gender, and emotion labels in real-time.    

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/c528d490-af23-4176-b7e4-022b2f4d6b96)

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/ba647fec-29fe-4baa-b352-b0b9d1f412fe)

3.2 Architecture of the various detections:
	
	In this architecture of the detection built on several components that work together to process of various design implemented to the inputs mentioned in the above dataset.

3.2.1 Architecture of age, gender emotion detection using input as image from dataset:
The design mentioned figure 17. begins by loading a dataset that contains labelled images. These labels include categories for age, gender. Once the data is loaded it undergoes preprocessing work this can involve steps like normalization where pixel values are scaled to a range that the model can work with effectively and data augmentation, which artificially increases the diversity of the dataset by making various alterations to the images system focuses on the head region of the images to improve the accuracy of the predictions, as facial features are essential for gender and emotion detection, and to some extent for age estimation as well Focusing on the head region, the system aims to refine the accuracy of predictions given the pivotal role of facial features in distinguishing gender and emotional states [Smith et al., 2022].
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/cf621ee9-e67a-4868-a81d-11fb0db02267)

Figure 17 Desing of the detection using input as image from dataset
In this stage, the system extracts labels from the dataset that will be used for supervised learning. Labels correspond to the correct output the model should predict, such as age range, gender, and emotion. For creating model using an architecture suitable for image analysis for various model such as CNN, YOLO and Wide Resnet. Then pre-trained model is used to detect emotion classifier separately.  The model is trained on the pre-processed images and their corresponding labels. During this phase, the model learns to associate the input data with the correct labels. The model's performance is evaluated using metrics such as mean absolute error for age prediction and accuracy for gender and emotion classification. After training, the model can predict the labels for new, unseen images. This involves determining the gender (male or female), estimating the age, and recognizing the emotion by using pretrained model (angry, fear, happy, neutral, sad, surprise) of the subjects in the images. Finally, the system outputs the predictions. For age, this might be a specific category or a range. Gender is typically a binary output, and emotion can be one of several categories that the system has been trained to recognize.

3.2.2 Architecture of age, gender emotion detection using Input as the video file
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/cff8c1ea-eace-4733-a128-f52abbf9a6c2)


In figure18. The system starts with video input, comprising a series of frames with subjects to be analysed for age, gender, and emotions. Initially, the system uses a face detection algorithm to identify faces within the video frames. This step narrows down the area of interest to facial regions, essential for accurate analysis. The WideResNet model, trained on a comprehensive dataset, is deployed for feature extraction, discerning intricate features from each face that correlate with age and gender. The extracted features are then processed by the WideResNet model, which predicts the age range and gender of each individual. The model’s architecture is specifically tuned to provide high accuracy in these domains. Concurrently, a separate pre-trained CNN model, optimized for emotion classification, analyses the same facial features to identify emotions. This CNN has been trained on a dataset labelled with various emotional expressions, enabling it to recognize emotions such as anger, fear, happiness, neutrality, sadness, and surprise with high fidelity. the predictions from both the WideResNet and the CNN models, the system ascribes a set of attributes to each face an age and gender category from the WideResNet and an emotion from the CNN. The final output includes labels for age, gender, and emotion for each detected face, superimposed on the video frames for real-time display or saved for subsequent analysis.

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/155f1ac2-d7d8-44de-9e81-8069e460ad97)
3.2.3 Architecture of age, gender emotion detection using real time video
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/dcff6597-1590-4885-b7a6-85e946839399)
The system takes in a surveillance or real-time video stream as its input source. The system utilizes a frontal face detector, which may refer to an algorithm like Dlib's frontal face detector, for the identification of faces in an image or video stream. This detector is specifically trained to recognize human faces from a front-on perspective, making it highly effective for capturing faces directly facing the camera. It's a critical first step in the process, ensuring that subsequent analyses, such as emotion classification using a CNN model, are performed on correctly identified facial regions within the video feed. The algorithm outputs detected faces, which are then used as the input for the subsequent analysis steps. The Wide ResNet (WRN) 16-8 model, a variant of Wide ResNet with 16 layers and a widening factor of 8, processes the detected faces to predict age and gender. WRN is known for its deep structure that's capable of learning highly discriminative features from facial data. CNN model is utilized to determine the emotional expressions on the detected faces. This CNN model, designed for real-time application, efficiently processes the facial features to classify emotions, benefiting from its capability to handle the intricacies and nuances of human expressions. Its architecture, potentially fine-tuned on a substantial corpus of emotional data, allows for precise recognition of a range of emotions, which is crucial for applications like sentiment analysis or interactive systems. The final step combines the predictions from both models. For each detected face, the system generates a bounding box annotated with the estimated age, identified gender, and the recognized emotional expression. This combined output is then likely displayed on the video feed or stored for record-keeping and further analysis.

3.3 Model: 
	The model selection for this project is one of the critical step of the pipeline in deep learning. From the literature review. In this project three working model gets implemented and done with comparative analysis is gone through with working model based on accuracy and mean absolute error best model get chosen for detecting the age and gender. The fourth model is emotion classifier to detect the emotion which it was pre trained model since training data is not allocated in the dataset. Thus, model get trained from the fer 2013 dataset where various facial emotion recognition is available.

3.3.1 Convolutional Neural Network
	
	The two CNN model constructed. First model focused on age detection, designed as a regression problem, while the second model targeted gender detection, formulated as a binary classification task. Both models employed a sequential arrangement of convolutional and max-pooling layers, followed by dense layers for final predictions. The datasets were pre-processed to suit the models' input requirements. Images were resized to 224x224 pixels and normalized for effective model training

Possible improvement for the base pipeline for the model were applied to enhance the robustness of the models against overfitting

•	Data augmentation techniques
•	including rotation,
•	width and height shifts
•	shear
•	zoom
•	flipping
•	Cross-validation

The age model was compiled with a mean squared error loss function, whereas the gender model utilized binary cross-entropy. The Adam optimizer was used for both models to ensure efficient training. For the gender model is to find has the classification problem to mention it has the accuracy level.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/7134c963-41e7-4386-b0ac-8500f696a9de)

The input shape is set to (224, 224, 3) which corresponds to images of size 224x224 pixels with 3 colour channels (RGB) which the UTK face dataset is approached with the cropped images with aligned specification. This layer has 32 filters, each of size 3x3.The 'relu' activation function introduces non-linearity, allowing the model to learn more complex patterns. Each filter convolves across the input image to detect various features like edges, textures, etc. This layer performs down-sampling by applying a 2x2 max pooling operation. It reduces the spatial dimensions (height and width) of the input volume for the next layer by half, which decreases the number of parameters and computation in the network, and helps prevent overfitting. Following the first convolutional and pooling layers are additional sets of these layers with more filters (64 and then 128).Each subsequent convolutional layer can detect higher-level features due to the increased complexity of the filters. Additional pooling layers continue to reduce the dimensionality of the feature maps. This layer flattens the 3D output of the previous layer into a 1D array without affecting the batch size.

It prepares the data for the fully connected layers that follow. This is a densely connected layer with 128 neurons. It takes the flattened input from the previous layer and learns non-linear combinations of features. This layer randomly sets half of the neurons in the previous layer to zero during training. It is a form of regularization to prevent overfitting by ensuring that the network remains robust and doesn't rely on any one neuron. This is the final layer for the age detection model, which has a single neuron since it's a regression problem. The model outputs a continuous value representing the predicted age.

The gender detection model has a similar architecture to the age detection model. These layers are identical in architecture to the age model, serving the same purpose of feature detection and dimensionality reduction. These layers function the same as in the age model, learning combinations of the features detected by the convolutional layers and preventing overfitting. This layer differs in that it has a sigmoid activation function, which is appropriate for binary classification problems. The result are tested separately in the another section. The models were compiled with the Adam optimizer, with the age model using mean squared error as the loss function and the gender model using binary cross-entropy. Training was performed over 100 epochs with checkpointing.

The input to the model is an image with a shape of 224×224 pixels with 3 channels corresponding to RGB colour space. The 32 Filters are used to extract various features from the input image. Each filter has a size of 3×3, which is standard for capturing the local context. The Rectified Linear Unit (ReLU) introduces non-linearity into the model, allowing it to learn more complex patterns. Pooling Size 2×2 This layer reduces the spatial dimensions (width and height) by half, which decreases the computation for the upcoming layers and helps in making the detection of features invariant to scale and orientation changes. The 64 Filter increases the capacity of the model to extract more complex features as it goes deeper. The same ReLU activation is used in here as like first one as same like maxpooling layer. 

For third layer 128 Filters uses Further increases the depth of the network, allowing the model to learn even more complex features. ReLU activation continues to add non-linearity. Continues to reduce the size and complexity of the feature maps for the third max pooling layer. Converts the 2D feature maps into a 1D feature vector, preparing the data for the fully connected layers. the first fully connected layer, or Dense Layer, comprises 128 neurons and serves as a pivotal component where the network begins to synthesize the extracted and flattened features into classifications. This layer employs the ReLU activation function to introduce non-linearity, essential for modelling the complex relationships within the data. Following this, a Dropout layer with a rate of 0.5 is implemented, which strategically deactivates half of the input units randomly at each training step, thereby mitigating the risk of overfitting by providing a form of regularization. The culmination of this architecture is the output layer tailored for age detection, consisting of a single neuron. This neuron's role is to output a continuous variable, reflecting the model's regression nature, which predicts age as a single value. Unlike the earlier layers, this output layer does not use an activation function, allowing for an unbounded numerical age prediction.

Data Augmentation	Value
Rotation Range	20
Width Shift Range	0.2
Height Shift Range	0.2
Shear Range	0.2
Zoom Range	0.2
Flip Horizontally	Yes
Fill Mode	nearest
Table 2 Data augmentation values used in the CNN model







Layer	Type	Output Shape	Param #
conv2d_1	Conv2D	(None, 224, 224, 32)	896
max_pooling2d_1	MaxPooling2D	(None, 112, 112, 32)	0
conv2d_2	Conv2D	(None, 112, 112, 64)	18,496
max_pooling2d_2	MaxPooling2D	(None, 56, 56, 64)	0
conv2d_3	Conv2D	(None, 56, 56, 128)	73,856
max_pooling2d_3	MaxPooling2D	(None, 28, 28, 128)	0
flatten	Flatten	(None, 100352)	0
dense_1	Dense	(None, 256)	25,690,368
dropout_1	Dropout	(None, 256)	0
dense_2	Dense	(None, 128)	32,896
dropout_2	Dropout	(None, 128)	0
dense_3	Dense	(None, 1)	129
Table 3 Modified YOLOv5 for this Project


The gender detection model shares the same architecture as the age detection model up to the first fully connected layer. Output Layer for Gender Detection 1 Neuron with Sigmoid Activation: Since gender classification is a binary classification problem, the sigmoid function is appropriate. It squashes the output to a probability range between 0 and 1, interpreting as the likelihood of the image being male or female.

3.3.2 YOLOv5 model for the pre-processing stage with combination of CNN layer

	The YOLOv5 architecture streamlines real-time object detection by integrating a backbone, neck, and detection head in a single forward pass. The backbone, often a CSPDarknet53, efficiently extracts features from input images, while the PANet neck further processes these features for multiscale detection. The head, with convolutional layers, predicts bounding boxes, objectness, and class probabilities. YOLOv5 operates on multiple scales, using anchor boxes to enhance detection across object sizes. Its unique loss function consolidates bounding box regression, objectness, and classification losses, refining training. The model is typically trained with backpropagation and optimizers like SGD or Adam and outputs detections post-inference. Due to its complexity, YOLOv5's architecture is not easily encapsulated in brief visuals but is thoroughly detailed in the official repository, offering insights into layer configurations and network depth (Ultralytics, 2021). Conv2D layers with increasing filter sizes (32, 64, 128) for feature extraction, using ReLU activation for non-linearity. MaxPooling2D layers following the convolutional layers to reduce the spatial dimensions of the feature maps (Goodfellow et al., 2016). Flatten layer to convert the 2D feature maps into a 1D vector. Dense layers, also known as fully connected layers, where the network learns to combine features to make predictions. It includes two Dense layers with 256 and 128 neurons, respectively, using ReLU activation.

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/83c34a85-ae3c-4062-a126-cc98e508f5ee)

Dropout layers are used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time, which helps to prevent overfitting. The final Dense layer with a single neuron without an activation function for regression output to predict the age. This is a binary classification task, where the model predicts a category (gender). The architecture is similar to the age prediction model with some differences

The final Dense layer uses a sigmoid activation function. This is because binary classification tasks require a sigmoid activation to predict the probability of the input being in one class or the other (male or female in this case). Both models are compiled with the Adam optimizer, which is an extension to stochastic gradient descent and is quite effective in practice. The loss functions are chosen according to the task: mean_squared_error for age prediction (regression) and binary_crossentropy for gender prediction (classification).

3.3.3 Wide Residual Network
	The wide_basic function defines a basic block for Wide ResNet. Each block consists of two 3x3 convolutional layers (conv1 and conv2). These convolutional layers are padded to maintain the size of the feature map and regularized with L2 regularization to mitigate overfitting. First convolutional layer with an optional stride.
Then Batch normalization is applied to stabilize learning and normalize the output of the previous layer. The model gains non-linearity from the ReLU activation function, which enables it to learn increasingly the patterns. Step size of the second convolutional layer is always (1,1). Following the second convolution, there is another batch normalization. A shortcut connection that applies a 1x1 convolution if the number of input planes (n_input_plane) doesn't match the output planes (n_output_plane), otherwise, it uses identity mapping. This helps in gradient flow and allows training of deeper networks. The result of adding the batch normalized output of the second convolution and the shortcut connection. From net The final output after applying a ReLU activation on the added tensor.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/1c0e2ef0-6e89-47a0-82c9-4b1c91eb9032)

The Input layer with shape input shape. Defines the number of filters for each stage of the network. The number of filters is scaled by the factor k. Initial convolutional layer before entering the wide residual blocks. The three block conveys block1, block2, block3 These are groups of wide residual blocks with increasing number of filters. Each group doubles the number of filters and reduces the spatial dimension by half using a stride of (2,2) in the first block of the group After the last block, another batch normalization and ReLU activation are applied. For pooling global average pooling reduces each feature map to a single value, which helps to reduce the model's parameters and computations. The pooled feature map is flattened to a vector to be fed into fully connected layers. Two separate output layers for the tasks—gender classification and age regression. The gender output uses a SoftMax activation for classification, and the age output doesn't use an activation function since it's a regression task.

The constructed Wide Residual Network (Wide ResNet) is adept at simultaneous classification and regression, optimizing the learning process for related tasks within a single model framework. This approach not only streamlines training but also has the potential to improve the model's ability to generalize across tasks. The distinctive characteristic of this architecture lies in its breadth - it employs an increased number of filters in each layer. This strategy allows the extraction of richer feature representations, eschewing the need for a deeper network, which can be advantageous in reducing computational demands and shortening training durations. The model is finely tuned to predict numerical age values and categorize gender, leveraging the expansive capacity of its residual blocks to effectively assimilate and interpret the input data without the complexity of an extended network structure.


3.3.4 Emotion classifier model (Pre-trained):
	The emotion detection system leverages a convolutional neural network (CNN) that processes facial images in grayscale. These images are first normalized and augmented, undergoing transformations such as rotation and shifts to enhance the robustness of the model against variations in new data. The CNN architecture is structured with a cascade of layers, each with a specific function

The Convolutional Layers Multiple layers with 32, 64, 128, and 256 filters capture various aspects of the input images. They act as feature detectors, picking up on edges, textures, and other facial details critical for determining emotions. The model employs the ELU activation function after each convolutional layer. ELU aids in introducing non-linear properties to the network, allowing it to learn complex patterns. Post-activation, batch normalization standardizes the inputs to the next layer, which helps in accelerating the training process and achieving stability. Spatial pooling reduces the feature map dimensions, distilling the image data into a more manageable form while preserving essential information. To combat overfitting, dropout layers nullify a portion of neurons randomly during the training phase, ensuring that no single neuron becomes too influential. The flattened layer transforms the pooled feature maps into a single vector that feeds into the dense layers, where deeper levels of learning occur. Here, the network combines features to form higher-level attributes that can be used for classification. A final dense layer with SoftMax activation computes the probability distribution over the six emotion categories, delivering the model's prediction. This model worked under the dataset of the FER 2013. It concerns the facial recognition and model gets work through several training and testing.

3.4 Training and Testing Data:

	The project employed a multi-faceted approach to training and testing models for age, gender, and emotion detection. A key aspect was ensuring that the models generalize well to new data, which was achieved through important data organization and validation techniques. Cross-Validation and bootstrap are popular algorithms that can be used for splitting the data into training, validation and test sets (Xu and Goodacre 2018).The datasets for both the 1500 and 2000 image models were organized into training, validation, and test sets. To the best practices observed in the literature, the images were not shared between the training and test sets to avoid bias and potential data leakage. The split was performed using a arranged sampling method, ensuring that the distribution of age and gender labels was consistent across the training, validation, and test sets. In order to improve the model's generalization capabilities, a K-fold cross-validation approach was employed. This method helps to ensure that the model is reliable by testing it on multiple unique data splits. For the age prediction model, this involved dividing the dataset into several subsets. The model was trained on one subset and then tested on another to determine the Mean Absolute Error (MAE). The same approach was used for the gender classification model, but with accuracy as the measure of performance.

The age prediction model's performance was gauged using mean squared error, while binary cross-entropy was used for the gender classification model. The training duration was determined by identifying the point at which additional epochs did not yield substantial gains in performance. Completion of training, the models were evaluated using the test set, which was never seen by the model during training or cross-validation. This step was crucial for assessing the models' performance and ensuring their applicability to real-world scenarios. From the table training and testing done under the two set of number of images 1500 and 2000 where 70% percentage for training, 15% for the validation set and 15 % for the testing set.

Set	Number of Images (for 1500 images)	Percentage (for 1500 images)	Number of Images (for 2000 images)	Percentage (for 2000 images)
Training Set	1,050	70%	1,400	70%
Validation Set	225	15%	300	15%
Test Set	225	15%	300	15%
Table 4 - Training, Validation and Test set organization.


When using the UTKFace dataset in your project, it's important to consider how the dataset's diversity and comprehensiveness impact the evaluation metrics. This research focused on age and gender estimation from face images, introducing the Audience benchmark for age and gender classification (Eidinger et al.2014)

3.4.1 Accuracy
The formula for calculating accuracy is:

Accuracy =  (Number of correct Predictions)/(Total Number of Predictions)

Accuracy in gender prediction would indicate how well the model performs across the diverse set of individuals in the dataset. Accuracy is a fundamental metric in machine learning and statistical modelling, especially for classification tasks which is majorly used in this project to calculate the accuracy level percentages for the gender classification. It provides a straightforward indication of a model's overall performance by comparing the number of correct predictions against the total number of predictions made. In your project on age, gender, and emotion detection, accuracy plays a crucial role, particularly in the gender classification aspect.


In mathematical terms:
Accuracy =(TP+TN)/(TP+TN+FP+FN)
Where:
	TP (True Positives): Number of correctly predicted positive observations.
	TN (True Negatives): Number of correctly predicted negative observations.
	FP (False Positives): Number of incorrect positive predictions.
	FN (False Negatives): Number of incorrect negative predictions.

3.4.2 Mean Absolute Error (MAE)
	From the mean absolute error is used metric for evaluating the accuracy of continuous variables in regression models. for this project the mae is used to calculate the error loss on particular age as regression model.

Formula:

The formula for calculating MAE is:

MAE = □(1/n) ∑_(i=1)^n |y_(true,i)-y_(pred,i) |


Where:

	y_(true,i  )represents the true values.

	y_(pred,i) represents the predicted values.

	n is the total number of observations
After training model to predict ages, you use a test dataset to evaluate its performance. This dataset contains images along with their true age values. Your model predicts an age for each image in the test set, resulting in a series of predicted ages. MAE is then calculated as the average of the absolute differences between the predicted ages and their corresponding true ages in the test set.
MAE in age prediction model serves as a clear and simple metric to understand the average error magnitude, thus providing insights into the typical amount by which the predicted ages differ from the actual ages. This can help in further refining the model or in setting expectations about its performance in real-world applications.


3.4.3 Precision, Recall, F1-Score:

Precision in Gender Classification involves determining the proportion of accurate predictions in all positive predictions. For example, if the model predicts 'male' 100 times and 90 of those predictions are correct, the precision is 90%. High precision in your project signifies that the model rarely makes mistakes in gender classification, which is crucial in scenarios where incorrect gender prediction could lead to significant implications. 

Precision = (True position(TP))/(TP+False Position(FP))

Recall (Sensitivity) in Gender Classification assesses the model's ability to identify all instances of a particular gender correctly. It's calculated by dividing the number of correct gender identifications by the total actual instances of that gender. 



For instance, if the model correctly identifies 80 out of 100 actual 'female' instances, the recall is 80%. This metric is essential in your project for ensuring that the model doesn't miss any true instances of a particular gender.

Recall =  (TP )/(TP+False Negatives (FN)) 

F1-Score in Gender Classification balances precision and recall, providing a singular metric that encapsulates both aspects of the model's performance. It's especially useful in your project for evaluating the model's overall efficacy in gender classification, especially when the dataset is imbalanced. Support in Model Evaluation shows the actual occurrences of each class in the test dataset. This metric is vital for understanding the context in which the model's performance metrics are evaluated. In your project, it helps in interpreting the model's performance by indicating how many instances of each gender were present in the test dataset.



F1 Score = 2 x (Precision x Recall )/(Precision+Recall)

3.5 Hardware Requirements:

To implement the practical part python programming language version 3.11.4 in local machine and 3.10.12 in google colab. In open-source library including Open Computer Vision, Pytorch and Keras have been employed. Further all experiments have been performed in google colaboratory environment and In PyCharm Community Edition 2023.2.3 is used as the virtual machine.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/dcae1e15-e398-4cfc-8b77-6daaeeee5004)


![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/544526d2-32dd-4029-b628-37bab15fb245)

Figure 23 and 24 represents the specification of the employed GPU and CPU on Google Colab. The mentioned version of the python and employed libraries in this study. The further code is uploaded in drive and the link is mentioned the implementation part.

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/a069d835-89ab-4565-bcc8-d58e82c2d962)

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/d44cb3a6-7678-4e4e-8ac5-90ba8e28e49a)
Figure 25 Version of the Python and various main libraries used in this project
4. Implementation:

		The source code and the implementation of all model used in this project has been uploaded in the one drive link including all weight file of model and dataset pictures. The all Coding part worked under the Google Colaboratory environment and for real time input has the video used under in the PyCharm community. The implementation phase of this project involved the integration of Convolutional Neural Networks (CNN), YOLOv5 and Wide ResNet, utilizing the UTKFace dataset. The project's codebase, primarily developed in Python, leverages the TensorFlow and Keras libraries for modeling and PyTorch for the YOLOv5 component. This hybrid approach capitalizes on TensorFlow's extensive ecosystem for deep learning applications and PyTorch's streamlined workflow for object detection tasks. For the CNN model, the project utilized the WideResNet architecture known for its performance in facial recognition tasks. The model was trained on images pre-processed to normalize and augment the data, enhancing the model's ability to generalize across different facial features and expressions. The YOLOv5 model, pre-trained on a comprehensive dataset, was fine-tuned with the UTKFace dataset, enabling the model to detect faces accurately before predicting age, gender, and emotion attributes.
The data augmentation pipeline employed techniques such as rotation, width and height shifts, shear, zoom, and horizontal flip to simulate a variety of real-world conditions. This was crucial in ensuring robustness and reducing overfitting, given the variability in real-world scenarios where the system would be deployed. During development, the Google Colab platform was instrumental, providing access to high-performance GPUs, which are essential for training deep learning models efficiently. This environment allowed for rapid experimentation with different hyperparameters and model structures without the constraint of local hardware limitations. The models were evaluated using standard metrics such as mean absolute error for age prediction and accuracy for gender classification. For emotion detection, a pre-trained model ‘emotion_little_vgg_2.h5 was employed, which was integrated into the system to work in tandem with the age and gender prediction models.

5. Result:
	This section is the result part obtained from the developed model that in part 3 described more in the details are presented. As mentioned, several characteristics of the model CNN, YOLO and Wide ResNet is used. Where first to test the sample input images to check the accuracy of the original image of age and gender. Then follow on the predicted emotion was mention as pre-defined model mentioned. After detecting age and gender the evaluation of the accuracy of the model implemented to dataset and mean absolute error is detected of each model by the loss variation. At last comparison 

5.1 Results based on Model CNN:
	In model CNN various stages of the output is experimented with different layer methods is used. 
5.1.1 Implementation of basic CNN model 
The age detection model, trained to minimize mean absolute error (MAE), achieved an MAE of 13.99598217010498 on the test set. This result indicates the average deviation of the predicted age from the actual age across all test samples. The gender detection model, optimized for classification accuracy, reached an accuracy of 0.6842105388641357 as mentioned in Table 5. This suggests that approximately 68% of the time, the model correctly identified the gender of the individual in the image. The Table 5. Result based on to the training and validation.
Metric	Value
Age Model Test MAE	13.99598217010498
Gender Model Test Accuracy	0.6842105388641357
Table 5 Result of accuracy and loss in CNN

The performance of the models can be contextualised by taking into account the task complexity, data diversity, and inherent limitations in age and gender prediction from photographs. The performance of the age model could be influenced by the wide diversity in physical ageing symptoms, whilst the gender model's accuracy could be influenced by the binary classification of a potentially non-binary feature. Future research could concentrate on tackling these complications, possibly by increasing the dataset or introducing more advanced features into the model. Importing sample input images to find age, emotion and gender detection 

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/a3bf1828-3a2f-4a5e-8c46-7225d810201a)
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/b216fa15-388d-49e7-8f29-b8e7bb82bc2b)
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/78db9190-188c-4135-90c7-ff5453477ab4)

	Precision
	Recall	F1-Score	Support
Male	0.89	0.86	0.88	658
Female	0.90	0.92	0.91	861
Accuracy			0.89	1519
Macro Avg	0.89	0.89	0.89	1519
Weighted Avg	0.89	0.89	0.89	1519

Table 6 Evaluation metrices

The evaluation of your gender classification model reveals an overall accuracy of approximately 89.47%, indicating that the model correctly identifies gender in most cases. Specifically, the model shows a slightly higher precision in identifying females (90%) compared to males (89%), suggesting it is more accurate in minimizing false positive gender identifications for females. In terms of recall, the model is more effective in recognizing all instances of females (92% recall) than males (86% recall), indicating a higher sensitivity in detecting females. The F1-Scores, which combine precision and recall, are robust for both genders, with 0.91 for females and 0.88 for males, showing a balanced performance in gender classification. The support numbers, 658 for males and 861 for females, reflect the actual occurrences of each gender in your test dataset, providing context to these metrics. The model's overall consistency and reliability are further underscored by the macro and weighted averages, both hovering around 0.89. These results highlight the model's effectiveness, with a slight edge in identifying female gender, and can guide future refinements.

5.1.2 Implementation of basic CNN model with higher image

The result analysis of the convolutional neural network (CNN) models for age and gender prediction shows promising performance. For the age model, the Mean Absolute Error (MAE) over the test set is 5.64, which implies that, on average, the age predictions are within approximately 5 to 6 years of the actual values. This indicates a reasonable level of accuracy, considering the challenging nature of predicting exact ages from images.

On the other hand, the gender model achieved an accuracy of approximately 73.54% on the test set. While not as high as might be desired, this does reflect a certain level of predictive power, especially given that gender classification can be complex due to the subtle differences that the model needs to learn from the training data.
Metric	Value
Age Model Test MAE	5.642977
Gender Model Test Accuracy	0.735417
Table 7 Result of accuracy and loss in CNN with higher images




The training and validation loss graphs for the age model display a typical convergence pattern, with training loss decreasing sharply and then plateauing, which is a sign of the model learning from the data effectively. The validation loss following a similar trend, but with some fluctuations, suggests the model is generalizing well without overfitting.
For the gender model, the training accuracy shows improvement over epochs, while the validation accuracy displays more variability. This could be a sign of the model struggling to generalize from the training data to unseen data, or it could indicate that the validation set has a slightly different data distribution than the training set. This model gone through 70 epochs for both age model loss and gender model accuracy.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/ce7765fe-5b1b-48d3-b5f2-39dfdbadb10d)



The validation steps in each epoch for the age model are consistent, with no spikes or dips indicating data anomalies or model instability. In contrast, the gender model's validation steps show some variation, which might warrant a closer examination of the data or the possibility of implementing techniques to improve model robustness.

Importing sample input images to find age, emotion and gender detection 
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/61262348-8591-4782-aab0-bebc042ce61b)
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/f2113eed-11fa-407b-8007-9113c6e163ca)

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/3074c0c9-315b-4ad3-8234-cec9b3b2536f)
The Age Model showcased a Mean Absolute Error (MAE) of 5.5588, suggesting that the model's age predictions were, on average, around five and a half years off from the actual ages. For the Gender Model, an accuracy of 82.06% was achieved, indicating a high level of reliability in gender classification.

	Precision	Recall	
F1-Score	Support
Male	0.84	0.70	0.76	663
Female	0.81	0.91	0.86	937
Accuracy			0.82	1600
Macro Avg	0.83	0.80	0.81	1600
Weighted Avg	0.82	0.82	0.82	1600
Table 8 Evaluation metrices for testing


In terms of precision and recall, the Gender Model had a higher precision (0.84) for males than females (0.81), meaning it was slightly more accurate when predicting males. However, it had a higher recall for females (0.91) compared to males (0.70), suggesting it was better at identifying all female instances in the dataset. The F1-Scores, which balance precision and recall, were 0.76 for males and a higher 0.86 for females, reflecting the model's better performance on female classifications. The support numbers show that there were 663 males and 937 females in the test set, providing the context for the model's performance metrics. These numbers demonstrate the model's capability to work with datasets that may have an uneven distribution of classes.

5.1.3 Implementation of CNN model with data augmentation
	 The Age Model yielded a Mean Absolute Error (MAE) of 9.7004, indicating that on average, the age predictions were about 9.7 years off from the actual values. For the Gender Model, the achieved accuracy was 71.88%, suggesting that it correctly identified the gender in approximately 72 out of every 100 predictions. In terms of training dynamics, the Age Model's training and validation loss showed convergence, meaning that the model was learning effectively without overfitting or underfitting significantly. As for the Gender Model, both training and validation accuracy improved over time, with validation accuracy tracking closely with training accuracy, which is a good sign of the model generalizing well.

Metric	Value
Age Model Test MAE	9.7004
Gender Model Accuracy	71.88%
Table 9 Result of accuracy and loss in CNN with augmentation
These outcomes can be seen as satisfactory considering the complexity of predicting precise ages from images and the good accuracy of differences in gender presentation. The age prediction model achieved a Mean Absolute Error (MAE) of 8.8427, which provides an average measure of the difference between the predicted and actual ages across the test dataset. Lower MAE values indicate more accurate age predictions.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/4f75a290-1021-4b5a-b45b-61633c5dd71a)

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/8894d9bb-b578-45b5-a9ce-7900d9b6d7b2)
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/5de3ee57-7ea4-4f3c-9784-33b2729004c7)
The gender prediction model attained an overall accuracy of 82.20%, reflecting its capability to correctly classify the gender in a majority of the cases. Looking at the precision and recall values, both male and female classifications have a precision of 0.82, indicating that the model is equally precise for both genders. However, the recall is slightly higher for females at 0.87 compared to males at 0.76, suggesting that the model is slightly better at identifying females than males.
            
	Precision	Recall	F1-Score	Support
Male	0.82	0.76	0.79	472
Female	0.82	0.87	0.84	590
Accuracy	-	-	0.82	1062
Macro Avg	0.82	0.82	0.82	1062
Weighted Avg	0.82	0.82	0.82	1062
Table 9 Evaluation metrices for testing

The F1-Score, which combines both precision and recall, is 0.79 for males and 0.84 for females, showing that the balance between precision and recall is slightly more favourable for female classifications. The macro and weighted averages both stand at 0.82, showing overall balanced performance across the genders. The 'Support' column indicates the number of true instances for each class in the test data, which is essential for understanding the dataset's class distribution.

5.1.4 Implementation of CNN model with k cross validation

The cross-validation process for the age model indicates a fluctuation in the mean absolute error (MAE) across different folds, with an overall mean MAE of approximately 9.81. The gender model's accuracy also shows variation with different folds, with an overall mean accuracy of around 0.66. This means that the gender model correctly predicted the gender of the individuals about 66% of the time across the different set of data used during cross-validation.

Fold	Age Model MAE	Gender Model Accuracy
1	Approx. 10.1	Approx. 0.62
2	Approx. 9.7	Approx. 0.66
3	Approx. 9.6	Approx. 0.70
4	Approx. 10.0	Approx. 0.66
5	Approx. 9.8	Approx. 0.68
Mean	9.81	0.663
Table 10 Evaluation MAE and accuracy using K-Fold cross validation

The table above showcases the model performance metrics for each fold of the cross-validation process. It provides a concise view of how the models' predictions varied across different segments of the data and what their average performance was like.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/1dae127a-63d6-400f-b9c5-23d423140c70)

For the age model, the graph titled "Cross-Validation MAE for Age Model" likely indicates how the mean absolute error (MAE) varies across different folds of the data. The y-axis represents the MAE, which quantifies the average magnitude of errors in a set of predictions, without considering their direction. Lower values of MAE indicate better performance, as they suggest smaller deviations from the true values. The x-axis represents the fold number in the cross-validation process.

The graph titled "Cross-Validation Accuracy for Gender Model" is likely showing the accuracy of the gender classification model across different folds. The y-axis represents accuracy as a proportion, ranging from 0 to 1 (or presented as a percentage). Higher values on this graph suggest better performance, with an accuracy of 1 (or 100%) being a perfect score where all predictions are correct. The x-axis, similar to the previous graph, represents the fold number.


5.2 Results based on Model CNN with YOLOv5:

The integration of YOLOv5 for face detection combined with a Convolutional Neural Network (CNN) for age and gender prediction showcases an innovative approach to classify and predict personal attributes from images. In your project, after preprocessing the images for YOLO and extracting faces, you trained separate CNN models for age and gender prediction. The age model, evaluated by Mean Absolute Error (MAE), yielded an MAE of approximately 16.88. This indicates that, on average, the age predictions deviated from the true ages by around 16.88 years. It's a quantitative measure that reflects the average magnitude of errors in a set of predictions, without considering their direction.

In contrast, the gender model, assessed by accuracy, achieved around 63.36% accuracy. This metric reflects the proportion of true results (both true positives and true negatives) among the total number of cases examined. A higher accuracy percentage is typically desired, indicating that a greater proportion of the model's predictions matched the true gender labels.

Metric	Value
Age Model Test MAE	16.87986
Gender Model Accuracy	63.36%
Table 10 Result of accuracy and loss in CNN with YOLOv5
The training process included data augmentation techniques such as rotation, shifting, shearing, zooming, and flipping, which can help improve model generalizability by introducing a wider variety of training samples. This is evident in the training and validation curves for MAE and accuracy, where the model's performance over epochs can be visualized. The training and validation MAE for the age model and the accuracy for the gender model were plotted over 30 epochs, showing how the model learned and adjusted its predictions over time. The plotted graphs depict the fluctuation and convergence of these metrics, providing insight into the model's learning behaviour throughout the training phase.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/a05a80c7-f160-4aaf-86e8-0cdd03bb2947)

These metrics provide a comprehensive assessment of the gender classification model's performance. The precision indicates the percentage of true positive predictions for each gender, while recall shows how many actual positives were correctly identified. The F1-score combines precision and recall into a single metric, providing a balanced view of each gender's classification performance. 


	Precision	Recall	F1-Score
	
Support

Male	0.49	0.91	0.63	510
Female	0.61	0.12	0.21	563
Accuracy	N/A	N/A	0.50	1073
Macro Avg	0.55	0.52	0.42	1073
Weighted Avg	0.55	0.50	0.41	1073
Table 11 Evaluation metrices for testing
The support column reflects the number of true instances for each gender in the evaluated dataset. The accuracy row, which typically contains the overall correct predictions over total predictions, is not applicable here per class and is shown only in the F1-score column for overall accuracy. The macro and weighted averages aggregate the performance across classes, with the weighted average accounting for class imbalance by weighting the metrics according to the number of true instances in each class.

5.3 Results based on Wide ResNet Model:
Wide ResNet multi-output model was constructed and trained to predict age and classify gender based on facial images. This model architecture is an advanced variant of deep learning models that extends the depth while keeping the computational efficiency in check. The dataset was sourced from a specified directory and pre-processed to match the input requirements of the model, with images normalized to a range between 0 and 1 for more effective learning. The gender labels were one-hot encoded to fit the binary classification output of the model. A train-test split was employed, setting aside 20% of the 
data for testing the model's performance after training. The model was then compiled with Adam optimizer, using categorical cross entropy and mean squared error as the loss functions for the gender and age outputs, respectively.

Metric	Value
Age Model Test MAE	10.87986
Gender Model Accuracy	63.38%
Table 12 Result of accuracy and loss in Wideresnet model
After training over 30 epochs with a batch size of 32 and using a validation split to monitor performance, the model achieved the following results on the test set:The total combined loss of the model was high, at 252.09, indicating the sum of losses from both outputs.The gender classification part of the model reported a loss of 0.952 and an accuracy of approximately 63.38%. This suggests that while the model can predict gender from images with some level of accuracy, there's significant room for improvement, potentially through more training data, hyperparameter tuning, or more complex model architectures.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/30d8d2d2-d8b5-4c90-8924-dc19dba77c82)
The age prediction output showed a mean absolute error (MAE) of about 10.74 years, which implies that on average, the age predictions deviated from the actual ages by this margin. While it reflects a basic predictive capability, the MAE suggests that the predictions are not particularly precise and could be improved. The model's training and validation accuracies for gender and MAE for age were plotted over epochs, indicating how the model's performance evolved during training.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/888e4a6e-fc73-46d5-926b-2ee502c31638)

![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/4a121a7f-881d-42b5-b8ce-af611dc3322d)
The evaluation of the age and gender classification models on a test dataset has yielded insightful metrics. The age model's mean absolute error is 11.83, suggesting that the age predictions deviate from the actual ages by nearly 12 years on average, indicating a need for model refinement. For gender classification, the model's accuracy stands at 62.43%, with precision rates of 60% for males and 64% for females, and recall rates of 45% and 76%, respectively. This demonstrates a stronger ability to correctly identify females than males. The F1-scores, which balance precision and recall, are 0.52 for males and 0.69 for females, highlighting a disparity in model performance across genders. The support numbers show that the dataset contains more females than males, which might influence the model's learning bias. Overall, while the gender model demonstrates a decent capability to discern gender, the results indicate there is significant room for improvement, particularly in enhancing the accuracy and recall for male predictions.

	Precision
	Recall	F1-Score	Support
Male	0.60	0.45	0.52	472
Female	0.64	0.76	0.69	590
Accuracy			0.62	1062
Macro Avg	0.62	0.61	0.60	1062
Weighted Avg	0.62	0.62	0.61	1062
Table 11 Evaluation metrices for testing


5.4 Input as video file predicting age, emotion and gender detection using Wide ResNet model:
This input video data based on the indication of human being who performing news reading around one-minute flash news as format of MP4 file and frame rate around 25.00 per second where this video file attached under the dataset drive.
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/194df7cc-c225-4ee4-8d08-271ee305e68d)
In Figure 36 we are using the PyCharm to run the code base on the model created using the dataset for the age and gender. The Age is detected as 34, gender has Female and emotion as happy. For emotion predicting it has the pre-defined model. Here it matches the accuracy level best at peak considering for detecting all the stages.

5.5 Input as real time video detecting age, emotion and gender using Wide ResNet model  
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/666c5d1c-2164-4c16-8164-9b9cb3fa6d61)

Figure 37 Detecting Age, emotion and gender detection in real time video

From Figure 37 By using real time live video For age prediction, the system defines several age ranges and associates the predicted age with one of these ranges to provide a user-friendly output, such as "(25-32)". The gender prediction component categorizes the subject as 'Male', while the emotion detection aspect classifies the facial expression into categories such as 'Happy'. Here it matches the accuracy level best at peak considering for detecting all the stages.


6 Project Management
In this chapter, we delve into the application of project management principles within the scope of this project. Initially, we examine Gantt charts for the clear understanding of the work done as planned. This is followed by an in-depth analysis of the project's risk management strategies and the measures taken to ensure quality throughout the project's lifecycle. Furthermore, we address the vital topics of data security and adherence to ethical standards in the conduct of this project.

6.1 Project Schedule
This project Schedule was started under the last week of November based on simplified Gantt chart. The Gantt chart divided into 8 task has listed in the graph. Figure 38 shows the planned schedule of overall project with ranges date mention from bottom to top Data collection to Report submission date and days has been mentioned. The more time took for the model development around 12 days
![image](https://github.com/Yogesh-653/Age-emotion-and-gender-detection-using-open-cv-and-deep-learning-with-images-and-real-time-video/assets/60870157/84379917-bed7-4ada-9f42-3e380aba0cab)
6.2 Risk Management

Effective risk management is pivotal in preventing potential setbacks and ensuring the smooth progression of any project. This machine learning and image processing project is no exception, having identified several risks that could impede its advancement. These include disruptions in model training, storage limitations, and data loss. To mitigate these risks, robust strategies have been proposed, ranging from leveraging cloud-based platforms to implementing regular data backups. Notably, Google Colab's complimentary GPU access has been instrumental in circumventing prolonged training durations, thereby sustaining the project's momentum. Below is a detailed table summarizing the identified risks, their mitigation strategies, and the potential impact if these risks were to materialize:

No.	Risk	Mitigation Strategy	Possible Solutions if Risk Materializes	Impact Level
1	Model Training Interruptions	Utilize persistent cloud services like Google Colab with session restoration.	Restart training using checkpointed models to resume from the last saved state.	Medium
2	Storage Constraints	Expand cloud storage or employ external drives for data redundancy.	Utilize additional cloud storage solutions or compress data to optimize space.	High
3	Data Loss	Implement automated, regular backups to multiple locations.	Restore data from the most recent backup. Employ data recovery tools or services if backups are corrupted.	High

During the project's lifecycle, a significant risk that materialized was an interruption in model training due to Google Colab's runtime disconnection. The impact was minimized by adopting checkpointing techniques, which allowed for the resumption of training without starting from scratch. Storage issues were addressed by allocating additional cloud storage and optimizing existing resources. To safeguard against data loss, a stringent backup protocol was enacted, ensuring that the latest project state could always be retrieved.

6.3 Quality Management
As strategy was maintained to follow the timeline of the project schedule as mentioned in grant chart. For doubt clearing and discussion throughout next step process has been done every fortnightly meeting with supervisor to get overall feedback. The meeting progress mentioned in appendix.

6.4 Social, Legal, Ethical and Professional Considerations

The photos and videos used for this project took from the publicly available data from the UTK Face dataset or any other dataset that contains images or video of individuals faces where it more publicly used. Then real time video is applicable for identifying the faces to detect age, emotion and gender detection and it not risk of re-identifying an individual of personal data of that human face is mentioned in the dataset. To prevent the public release of their data, data protection measures were implemented. The drive is only accessible to authorized personnel only for supervisors. Every piece of supplied data was stored on a Coventry University cloud platform that required a password to access. Ethics application for this project was approved and it is attached as Appendix. 

7   Critical Appraisal

This project delves into the sophisticated realm of age, gender, and emotion detection by harnessing the capabilities of Convolutional Neural Networks (CNN) and YOLOv5, in conjunction with the extensive UTKFace dataset. The innovative approach combines advanced deep learning techniques to enhance the precision with which machines interpret human facial characteristics.

One of the core strengths of this attempt is the dual-structured architecture. The CNN component is adept at extracting intricate image features, a process further optimized by the WideResNet model, known for its ability to learn more abstract representations. Meanwhile, YOLOv5 contributes its rapid object detection capabilities, ensuring quick and efficient performance. This powerful combination is bolstered by the UTKFace dataset's broad demographic representation, offering a rich spectrum of age, ethnicity, and gender data, which lays a robust groundwork for the model's training. The integration of data augmentation in the preprocessing phase expands the model's exposure to an array of facial orientations and expressions, significantly enhancing its real-world application adaptability. However, the project is not without its limitations. The UTKFace dataset, despite its diversity, presents limitations in representational diversity and sample size when accounting for the global population's variance. The computational demands posed by the CNN-WideResNet-YOLOv5 architecture may also restrict deployment on less powerful devices.

From an innovation perspective, this project's multi-attribute detection capability is its unique selling point, charting a relatively untrodden path compared to studies that concentrate on single attributes. The inclusion of YOLOv5 ensures expedited detection without undermining the CNN-WideResNet's meticulous feature extraction process, thereby contributing valuable advancements to the field. In the context of existing literature, this project sets itself apart through the deployment of a hybrid model that simultaneously extracts multiple facial features, utilizing a methodology that remains largely unexplored. The employment of the UTKFace dataset solidifies the project's dedication to a comprehensive and inclusive facial analysis approach.

Future iterations of the project could further the investigation into lightweight neural network architectures to broaden access across various devices. Moreover, a thorough exploration into the ethical dimensions of facial recognition technology is imperative, especially concerning privacy and consent in data utilization.


8    Conclusions
The chapter provides an overview of the project dissertation's accomplishments and provides a quick overview of some intriguing subjects for further study. It is determined that the project's attempt to address the successful research topic was “How a state-of-the-art Age, emotion and gender detection done with deep learning algorithm and how computer vision works when limited and low-quality input images and video are available?”. The main lesson selection of the suitable algorithm to work with the model and low quality, data limitation the computer vision uses the change the frame work to have clear picture and the accuracy when RGB method used to implement in the model.

8.1 Achievements
Every goal outlined in the first chapter was accomplished. According to the literature review CNN, YOLO and Wide ResNet model used to implement this project. According to this the best model with the with comparative analysis to detect the age emotion and gender detection is CNN with data augmentation by including higher images for the training process which give the accuracy of 73%. the model achieved a high degree of accuracy in identifying diverse facial attributes. Key accomplishments include the creation of a dynamic model that not only detects age and gender with high accuracy but also discerns between different emotional states. The model's performance is indicative of its potential impact on fields such as human-computer interaction, security, and personalized content delivery. Data augmentation techniques played a crucial role in enhancing the model's generalization capabilities, enabling it to handle real-world variations in facial data effectively. The inclusion of YOLOv5 in the project's framework significantly improved object detection speeds, which, when combined with the depth of CNN and the efficiency of WideResNet, resulted in a real-time processing capability that is both fast and reliable. The project's model distinguished itself from standard applications by demonstrating an ability to process and analyse multiple facial attributes simultaneously, which is a step forward in the realm of facial recognition technology.
8.2 Future Work
For future enhancements of the project, several avenues can be explored to address existing limitations and expand the model's capabilities. One of the primary areas of focus could be the expansion of the dataset. While the UTKFace dataset provides a broad range of facial images, incorporating additional datasets or using techniques like Generative Adversarial Networks (GANs) could generate more diverse training samples, especially for underrepresented demographics or subtle emotional expressions and the more accurate detail of the Age and Gender. Further refinement of the model could involve integrating more advanced neural network architectures or exploring the synergies between different models to improve accuracy and speed. For instance, experimenting with different configurations of WideResNet layers or adapting the YOLOv5 framework to be more sensitive to the nuanced features of age, gender, and emotion could yield better results.









