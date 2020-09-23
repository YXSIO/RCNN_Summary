# RCNN_Summary

This is the summary of the KaiKeBa course R-CNN, fast RCNN and faster RCNN. Here is the [link]() for the details of the RCNN related algorithms. 

## Table of contents
   * [Course summary](#course-summary)
   * [Regional CNN](#Regional-CNN)
      * [Basic idea](#Basic-idea)
      * [Four major components](#Four-major-components)
      * [To improve](#To-improve)
   * [Fast RCNN](#Fast-RCNN)
      * [To improve](#To-improve)
      * [Single number evaluation metric](#Single-number-evaluation-metric)
   * [Faster RCNN](#Faster-RCNN)
   	  * [To improve](#To-improve)
      
## Course summary
> You will learn how to build a detection system based on classifier. For further improvement, research used to focus on efficiency(Yolo), as well as accuracy.
> The Trios of RCNN will help you to understand how the detection algorithm innovate in order to improve the accuracy and efficiency.
> Further, to understand how attention works in CV is another aspect to make your CV knowledge to a new level. 

## Regional CNN
### Basic idea
- To identify(SVM) and modify(regressor) the candidates regions(SS). 
- Sliding window + classifier: The detection system is built on classifier. However, this design is very computationally intensive. 

### Four major components
- Selection search
	1. Selection search: segmentation -> greedy algo: similarity comparison among neighbours
	2. SS mainly find the 2000 candidate bounding boxes. 
	3. How to measure the similarity of each region: 
		- The feature for each region consists of color, texture and size which are measured by histogram.  
	
- CNN feature extraction
	1. Alexnet is used as the classifier to extract CNN features. 
	2. CBRP32: convolution, batch normalization, relu and pooling with 32 channels. 
	3. Feature extraction: 2000 computation to calculate the Pool5 and FC7 for each candidate regions. 
	
- SVM classifier
	1. Training the classifer involves nagative and positive data preparation.
		- The ratio is 1:3, since positive is better defined then those negative samples. 
	2. From the 2000 candidates, if there are 3 classes. Then 2000*3 SVM classification.
		- As a result, there are 100 heads, hulu 88 and mountain 45. 
	3. Then apply NMS to further eliminate the number: head 7, hulu 7 and mountain 5.
	4. NMS: How to reduce the redundant bb? 
		- Find the most confident bb with the highest confidence. 
		- Then suppress the BBs who overlaps with the select max. 
		- Iterative over the rest of the BBs. 
	
- Bounding box regression
	1. Detection: boundiung box location: tranverse all the location and size of the bounding boxes.
	2. Finding good bb candidate could improve the efficiency 
	3. Different class correspond to different LR, since each class has different feature maps.
	4. For each class, use the extracted feature as x and train four different LR. 
	5. Why use pooling5? Answer: pooling maintain the relative pixel relationship. 
	6. The central element of Pooling 5 has nearly global view for the original image. 

### To improve
- Slide window will be replaced by selective search and classifier will be replaced by CNN feature and SVM.

## Fast RCNN

### Key idea
``` 
- 联合计算：将各个component combine together.
	- Combine CNN and SVM together through VGG
	- Jointly training LR and Classification together via combine the CE loss with the smooth L1 loss.
 ```
- ROIPooling will sample image in different size to the same output. Although ROIPooling will lose information, multiple channel pooling will minimize such lose. 

### To improve
- Based on the 2000 candidate regions, try to improve the computation
- Through ROIPooling, combine CNN, SVM and regression
- Reduce the CNN computation: extract the region on the final feature map of the entire image. However, each image need size normalization before training. 
	1. Design a flexible pooling to resize the image to the same size. 
	2. Implement through ROI pooling: sampling. 
- Cannot reduce the number of candidate e.g.: 2000. 

### Details:
- **VGG** replace the previous AlexNet
- Image -> **VGG**, the size shrink while the location remains. Therefore, extract the 2000 bb out of the feature maps of that image before applying ROIPooling. 
- Use IOU to identify the label of the candidate bb. Not all the 2000 are used for training. 
- Label: class for each bb and its tx, ty, tw, th.
	- Smooth L1 VS L1: it is not derivative when x = 0, which make the solution very unstable.
	- L1 VS L2: L1 loss for location, less sensitive to outliers.
- Only 64 regions are selected from each image. 
- Batch = 128 = 64*2. The ration of positive and negative samples are 1:3. 
- Use the above dataset to fine tune the VGG.
- Through experiment, the performance of VGG over perform VGG + SVM. It makes sense since VGG is better at extracting image features. 

### Recap the forward and backward calculation of Fast RCNN

## Faster RCNN
### Key idea
- Deep convolution generate pixels which represent different size of region in the original image. 
- Convolution could be treated as moving window classification: One time convolution is equivalent to many times of sliding window classification. 
- Each classification header represent a anchor box with certain size, since the shape of anchor box is a parameter in the classification header, which could be learnt when given the correspond label. 

### RPN
- Insert two extra layers of convolutions to Integrate SS into CNN process.
	- 3 X 3 X 256 is the first convolution layer
	- Assuming 9 anchor boxes: 256 X 1 X 1 X 18 for classification and 256 X 1 X 1 X 36 (4X9) for bounding box regression.
- The final output of RPN is the candidate boxes. These boxes will be used to extract regions in the feature maps of the original image. 
RPN is two stages since it first replace ss and then apply regression and classification again to generate the final output. 

### Anchor
- Classification header consists of 6 Conv1X1
- Each pixel in the final feature map corresponds to a anchor box in the original image. 
- The final box candidate = total number of pixel * 6. E.g.: raw image 400 by 400. Sampling rate 16. The final feature map shape is 25 by 25. 


### Training process
1. Training VGG and fine tune with selected positive and negative labels. 
2. VGG + RPN
3. VGG + RPN + Classification header to perform finer classification and regression.
4. Iterate over to fine tune RPN and header with (LR=0)

### Data preparation for training
1. VGG: ImageNet 1000 class data
2. RPN: Select candidate window from sliding window and modify the location of candidate to output as ROIs.
	- The loss is still adjoint loss between softmax loss and smooth L1 loss.  

<br><br>
These Notes were made by [Yuxiang Zhang](mailto:kimiyuxiang@gmail.com) @2020
