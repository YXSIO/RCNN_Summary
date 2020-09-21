# RCNN_Summary

This is the summary of the KaiKeBa course R-CNN, fast RCNN and faster RCNN. Here is the [link]() for the details of the RCNN related algorithms. 

## Table of contents

* [Structuring Machine Learning Projects](#structuring-machine-learning-projects)
   * [Table of contents](#table-of-contents)
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
	1. Design the bb candidate: Selection search: segmentation -> greedy algo: similarity for neighbours
	2. SS mainly find the 2000 candidate bounding boxes. 
	3. How to measure the similarity of each region: 
		- The feature for each region consists of color, texture, size which are measured by histogram.  
	
- CNN feature extraction
	1. Alexnet is used as the classifier to extract CNN features. 
	2. CBRP32: convolution, batch normalization, relu and pooling with 32 channels. 
	3. Feature extraction: 2000 computation to calculate the Pool5 and FC7 for each candidate regions. 
	
- SVM classifier
	1. Training the classifer involve nagative and positive data preparation
	2. From the 2000 candidates, if there are 3 classes. Then 2000*3 SVM.
		- As a result, there are 100 heads, hulu 88 and mountain 45. 
	3. Then apply NMS to further eliminate the number: head 7, hulu 7 and mountain 5.
	4. NMS: How to reduce the redundant bb? 
		- Find the most confident bb with the highest confidence. 
		- Then suppress the BBs who overlaps with the select max. 
		- Iterative over the rest of the BBs. 
	
- Bounding box regression
	1. Detection: boundiung box location: tranverse all the location and size of the bounding boxes
	2. Finding good bb candidante could improve the efficiency 
	3. Different class correspond to different LR, since each class has different feature maps.
	4. For each class, use the extracted feature as x and train four different LR. 
	5. Why use pooling5? Answer: pooling maintain the relative pixel relationship. 
	6. The central element of Pooling 5 has nearly global view for the original image. 

### To improve
- Slide window will be replaced by selective search and classifier will be replaced by CNN feature and SVM.

## Fast RCNN

### Key idea
- ``` 联合计算：将各个component combine together. ```
- ` 联合计算：将各个component combine together. `

### To improve
- Based on the 2000 candidate regions, try to improve the computation
- Through ROIPooling, combine CNN, SVM and regression
- Reduce the CNN computation: extract the region on the final feature map of the entire image. However, each image need size normalization before training. 
	1. Design a flexible pooling to resize the image into the same size. 
	2. Implement through ROI pooling: sampling. 
- Cannot reduce the number of candidate e.g.: 2000. 

### Details:
- VGG replace the previous AlexNet
- Use IOU to identify the label of the candidate bb. Not all the 2000 are used for training. 
- Only 64 regions are selected from each image. 
- Batch = 128 = 64*2. The ration of positive and negative samples are 1:3. 
- Use the above dataset to fine tune the VGG.
- Through experiment, the performance of VGG over perform VGG + SVM. It makes sense since VGG is better at extracting image features. 



### Recap the forward and backward calculation of Fast RCNN

      |                | Predicted cat  | Predicted non-cat |
      | -------------- | -------------- | ----------------- |
      | Actual cat     | 3              | 2                 |
      | Actual non-cat | 1              | 4                 |
  - **Precision**: percentage of true cats in the recognized result: P = 3/(3 + 1) 


- So we can solve that by choosing a single optimizing metric and decide that other metrics are satisfying. Ex:
  ```
  Maximize F1                     # optimizing metric
  subject to running time < 100ms # satisficing metric
  ```

  - `OldMetric = (1/m) * sum(y_pred[i] != y[i] ,m)`
    - Where m is the number of Dev set items.

## Faster RCNN
### RPN
### Anchor

- Whereas in transfer learning, you have a sequential process where you learn from task A and then transfer that to task B. In multi-task learning, you start off simultaneously, trying to have one neural network do several things at the same time. And then each of these tasks helps hopefully all of the other tasks. 
- Example:
  - You want to build an object recognition system that detects pedestrians, cars, stop signs, and traffic lights (image has multiple labels).
  - Then Y shape will be `(4,m)` because we have 4 classes and each one is a binary one.
  - Then   
  `Cost = (1/m) * sum(sum(L(y_hat(i)_j, y(i)_j))), i = 1..m, j = 1..4`, where   
  `L = - y(i)_j * log(y_hat(i)_j) - (1 - y(i)_j) * log(1 - y_hat(i)_j)`
- In the last example you could have trained 4 neural networks separately but if some of the earlier features in neural network can be shared between these different types of objects, then you find that training one neural network to do four things results in better performance than training 4 completely separate neural networks to do the four tasks separately. 
- Multi-task learning will also work if y isn't complete for some labels. For example:
  ```
  Y = [1 ? 1 ...]
      [0 0 1 ...]
      [? 1 ? ...]
  ```
  - And in this case it will do good with the missing data, just the loss function will be different:   
    `Loss = (1/m) * sum(sum(L(y_hat(i)_j, y(i)_j) for all j which y(i)_j != ?))`
- Multi-task learning makes sense:
  1. Training on a set of tasks that could benefit from having shared lower-level features.
  2. Usually, amount of data you have for each task is quite similar.
  3. Can train a big enough network to do well on all the tasks.
- If you can train a big enough NN, the performance of the multi-task learning compared to splitting the tasks is better.
- Today transfer learning is used more often than multi-task learning.

### What is end-to-end deep learning?

- Some systems have multiple stages to implement. An end-to-end deep learning system implements all these stages with a single NN.
- Example 1:
  - Speech recognition system:
    ```
    Audio ---> Features --> Phonemes --> Words --> Transcript    # non-end-to-end system
    Audio ---------------------------------------> Transcript    # end-to-end deep learning system
    ```
  - End-to-end deep learning gives data more freedom, it might not use phonemes when training!



<br><br>
<br><br>
These Notes were made by [Yuxiang Zhang](mailto:kimiyuxiang@gmail.com) @2020
