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

Here are the course summary as its given on the course [link](https://www.coursera.org/learn/machine-learning-projects):

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
	2. Finding good bb candidante could improve the efficiency 
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
- Integrate SS into CNN end to end process


### RPN

### Anchor

### Multi-task loss


<br><br>
<br><br>
These Notes were made by [Yuxiang Zhang](mailto:kimiyuxiang@gmail.com) @2020
