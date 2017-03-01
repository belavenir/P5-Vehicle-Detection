##Vehicle Detection Project

---


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/1.png
[image4]: ./output_images/2.png
[image5]: ./output_images/3.png
[image6]: ./output_images/4.png
[image7]: ./output_images/5.png
[image8]: ./output_images/6.png
[video1]: ./result.mp4

---


###Histogram of Oriented Gradients (HOG)

####1. Extract HOG features from the training images

The code for this step is contained in the 6th code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Choose the HOG parameters

After trying various combinations of parameters, I think these parameters are suitable with respect to "detection task" and "computation cost". From the above test image, we can clearly tell the car shape from HOG feature map in a low caculation time.

####3. Train a classifier


I trained a linear SVM on the extracted features (color_histogram feature + HOG feature). The features are normalized. The training code can be found in the 9th code cell in Ipython Notebook. The test accuracy of final traing model is 0.9941. It is close to training accuracy.


###Sliding Window Search

####1. Implement Sliding window Search

I used the method proposed in class. The code can be found in the 12th code cell of Ipython Notebook. For time saving, I extract all the feature from a whole image once instead of sliding windows. The treshold heatmap and 'label' function is used for search best vote of detected car.

In my initial pipeline, the result looks good but I think it is better to set up multiple scale sliding windows with respect to different image area. I set up minimal scale in horizonal area, maximum scale for nearby area and a intermediate scale between them. In this way, I build up the initial pipeline(see the 14th code cell) with 3 different scale of windows. Moreover, I use a threshold to filter small detetected windows when `x-man-xmin < 50` or `yman-ymin < 50` as coded in the 13th cell. I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

####2. Final Pipeline

From the above result, we can still find some false positive. As for a video, we can benefit from mutiple frames to filter false negative. Because a detection should be found at or near the same position in several subsequent frames. The simplest way to do this is to store the most recent n number of heat maps, take a sum of all of the heat maps, and set a threshold on the combined heat map. In the test I choose 10 subsequent frame to test. The code is available in the 16th and 17th cell.

---

### Video Implementation

####1. Final video output. 

Here's a [link to my video result](./result.mp4)


####2. Filter of False Positive

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

In the video test, I also use multiple frames to filter noises as I mentioned before. And it works well.
---

###Discussion

In the result, we can see somewhat wobbly or unstable bounding boxes occur sometime, especially under tree shade. I tried to tune my parameter but it is difficult to remove. I think maybe we can agument some other dataset by changing illumination to make the training model more robust. Meanwhile a CNN model deserve a try as well. 
