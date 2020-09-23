# Zero Shot Sketch-Based Image Retrieval

The problem of retrieving images from a large-database using ambiguous sketches has been addressed. This problem has been addressed in the **zero-shot scenario**, where the test sketches/images are from **unseen classes** and the deep feature extractor's output embedding distances of the sketch and the image have been used to retrieve the top k closest images from the image database of the unseen classes.

# Results on unseen(zero-shot) classes

The below table presents a few qualitative results of our model on unseen test classes. The table presents the top 5 results from left to right.
  
| Query Sketch 1  |
|:---------------:|
|![](docs/examples/1/n03512147_1442-5.png)|

| Retrieved Image 1 | Retrieved Image 2 | Retrieved Image 3 | Retrieved Image 4 | Retrieved Image 5 |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
|![](docs/examples/1/ext_620.jpg)|![](docs/examples/1/ext_479.jpg)|![](docs/examples/1/ext_441.jpg)|![](docs/examples/1/ext_437.jpg)|![](docs/examples/1/n03512147_44302.jpg)|

| Query Sketch 2  |
|:---------------:|
|![](docs/examples/2/n02958343_10092-1.png)|

| Retrieved Image 1 | Retrieved Image 2 | Retrieved Image 3 | Retrieved Image 4 | Retrieved Image 5 |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
|![](docs/examples/2/ext_201.jpg)|![](docs/examples/2/n02958343_13615.jpg)|![](docs/examples/2/n04166281_6690.jpg)|![](docs/examples/2/ext_389.jpg)|![](docs/examples/2/n04166281_241.jpg)|


# Architecture Overview

# Instructions
<details>
<summary>
Installation
</summary>

Please execute the following command to install the required libraries:

```
pip install -r requirements.txt
```

</details>
<details>
<summary>
Data
</summary>

Please download and extract this file:

[The Sketchy dataset](http://transattr.cs.brown.edu/files/aligned_images.tar) - 1.8 GB

</details>
<details>

<summary>
Training
</summary>

</details>

<details>

<summary>
Inference
</summary>
</details>


