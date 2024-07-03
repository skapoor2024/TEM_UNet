# TEM_UNet

## Goal of this project

Our lab developed a neural network model to predict the behavior of grains in polycrystalline structures. However, the model was limited by its reliance on simulated data produced by SPPARKS , a Monte Carlo based model.
To address this limitation, this project focused on creating a image segmentation system that could identify individual grains within images captured using Transmission Electron Microscopy (TEM) to generate real world data. By incorporating real-world data from TEM images, we hoped to verify the algorithms integrity for the real world dataset. 

## Process and Solution 

Accurate segmentation of individual grains in Transmission Electron Microscope (TEM) images is crucial for analyzing real-world materials. To achieve this, I developed a process that combines supervised learning with traditional image processing techniques.

The core approach utilized a combination of the watershed algorithm and a UNet model. The watershed algorithm provided an initial segmentation for the grains, generating "pseudo labels." These labels were then used to train a UNet model, a type of neural network known for its effectiveness in image segmentation tasks. The UNet architecture was specifically chosen because of its ability to capture the fine details and contextual information present in TEM images, which are essential for accurate grain identification.

Following the initial segmentation by UNet, I applied a series of post-processing techniques to further refine the results.Morphological operations like dilation and skeletonization were used to connect fragmented grain segments and improve the accuracy of the boundaries. Additionally, a graph-based algorithm was employed to further refine the segmentation by treating junction points as nodes and extracting individual grains for precise boundary identification.
﻿
Beyond the core approach, I also explored incorporating temporal consistency, particularly when dealing with multiple TEM image frames. This technique, called "mode filtering," leveraged the assumption that grain sizes remain relatively consistent over short time sequences. By analyzing consecutive frames and employing a "voting system" based on grain presence in each frame, this method aimed to improve the overall consistency and reduce noise in the final segmentation.

## Outcome of the project

