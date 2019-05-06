# Application of CNN-V for Near Duplicate Video Retrieval

Anuja Golechha, Shivanee Nagarajan, Varun Ramesh

May 6, 2019

## Introduction

In this project we have performed near duplicate video retrieval/detection (NDVD) using a CNN based approach. With several state-of-the-art approaches available such as Multi-view hashing, Pattern-based approach, ACC, and CNN-based models Motivated by the use of CNN's over a wide variety of computer vision problems, we leverage the power of a pre-trained neural network to generate a global-level video histogram by means of vector aggregation over smaller components. We have modified the problem of NDVR into a model of querying a video database. In essence, the problem statement is - Given a dataset of videos and a query from a new video, we aim to retrieve a ranked list of near-duplicate videos. 

## Problem Domain
The problem of Near duplicate video retrieval is a sub-problem of the duplicate videos domain. In the latter, two videos are exact duplicates and hence, techniques that work at extracting and comparing pixel-by-pixel equality or signatures of keyframes work very well. The former class (chosen for this project) allows for format - scale, encoding, transformations or content modifications - watermarks, lightning, color changes. Hence, this model requires further featurization and deep net techniques for efficient detection.

## Significance of Near Duplicate Video Retrieval

The growth of video content has grown explosion exponentially over the last decade. With more than three-quarters of internet traffic being driven by video content and several large companies hosting video-sharing networks, it has become a core component of the internet world today. An interesting point of note is the high degree of near duplicates that have observed in this traffic.

Owing to this factor, companies need to build robust systems to detect and avoid storing duplicates (or certain video content) based on their policies. All of the above has spurred interest towards and NDVR, and, we have attempted to utilize this opportunity and build one such robust system.

## Exploration of Keyframe extractors
A video can be split into scenes, shot, and frames as shown in the following diagram. 
! [Image] (https://github.com/VarunCode/Application-Extension-of-CNN-V-for-Near-Duplicate-Video-Retreival/blob/master/keyframe.png)

In this project, we explore several keyframe extractor techniques as they serve as input to the CNN model. We also trade-off the performance and runtime of said techniques to provide a situational analysis of when to pick a particular model. Following is a summarization of some common techniques:

1. Sequential Comparison between frames - compare frame by frame and choose frames with high difference values.

2. Global Comparison between frames - Minimize a predefined objective function by using the global difference between frames

3. Curve Simplification - Represent frames as points in the feature space to formulate a trajectory curve. Key frames are the points that give the best representation to the shape of the curve.

4. Clustering - Each frame is a data point in the feature space. Cluster frames, then the frames that have smallest distances (closest) to cluster centers are selected.

We have picked method 1 - sequential comparison for our exploration and have implemented three subtechniques under this domain - local maxima, top k, and threshold parameter based methods.

