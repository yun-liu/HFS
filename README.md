## HFS: Hierarchical Feature Selection for Efficient Image Segmentation

Created by Yun Liu at Nankai University

### Introduction:

<img src="http://mmcheng.net/wp-content/uploads/2016/10/HFS_sample.png" width="800">

We propose a real-time system, Hierarchical Feature Selection (HFS), that performs image segmentation at a speed of 50 frames-per-second. We make an attempt to improve the performance of previous image segmentation systems by focusing on two aspects: (1) a careful system implementation on modern GPUs for efficient feature computation; and (2) an effective hierarchical feature selection and fusion strategy with learning. Compared with classic segmentation algorithms, our system demonstrates its particular advantage in speed, with comparable results in segmentation quality. Adopting HFS in applications like salient object detection and object proposal generation results in a significant performance boost. Our proposed HFS system can be used in a variety computer vision tasks that are built on top of image segmentation and superpixel extraction. Detailed description of the system can be found in our [paper](http://mmcheng.net/hfs/).

### Citations

If you are using the code provided here in a publication, please cite our paper:

    @inproceedings{cheng2016hfs,
      title={HFS: Hierarchical Feature Selection for Efficient Image Segmentation},
      author={Cheng, Ming-Ming and Liu, Yun and Hou, Qibin and Bian, Jiawang and Torr, Philip and Hu, Shi-Min and Tu, Zhuowen},
      booktitle={European Conference on Computer Vision},
      pages={867--882},
      year={2016},
      organization={Springer}
    }
    
    @conference{liu2018deep,
      title={DEL: Deep Embedding Learning for Efficient Image Segmentation},
      author={Yun Liu and Peng-Tao Jiang and Vahan Petrosyan and Shi-Jie Li and Jiawang Bian and Le Zhang and Ming-Ming Cheng},
      booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
      year={2018}
    }

### Installation

1. Clone the HFS repository
    ```Shell
    git clone https://github.com/yun-liu/hfs.git
    ```
  
2. Run Matlab script `dataset/cvt_format_bsds.m` to convert BSDS500 dataset to our C++ format

3. Unzip the `dataset/model.zip` package, and put all files into `/path/to/BSDS500/Results/` folder. It should have this basic structure

    ```Shell
    $BSDS500/
  	$BSDS500/Results/mergePartial.idx
  	$BSDS500/Results/mergePartial.rlt
  	$BSDS500/Results/Segment.WS
  	# ... and several other directories ...
    ```

4. For windows code, we tested with CUDA 7.5, OpenCV 3.0.0 and visual studio 2013.
   For linux code, we tested with CUDA 8.0 and OpenCV 3.0.0 under ubuntu 16.04.

5. Modify the path in `kernel.cu` file to your path (`/path/to/BSDS500/` or `/path/to/image`).

Now, you can run it. Have fun!

### Acknowledgment

This code is based on gSLIC. Thanks to the contributors of gSLIC.

    @article{gSLICr_2015,
        author = {Carl Yuheng Ren and Victor Adrian Prisacariu and Ian D Reid},
        title = {gSLICr: SLIC superpixels at over 250Hz},
        journal = {ArXiv e-prints},
        eprint = {1509.04232},
        year = 2015,
        month = sep
    }
