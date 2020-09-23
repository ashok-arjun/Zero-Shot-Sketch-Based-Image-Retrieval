# Zero Shot Sketch-Based Image Retrieval

The problem of retrieving images from a large-database using ambiguous sketches has been addressed. This problem has been addressed in the **zero-shot scenario**, where the test sketches/images are from **unseen classes** and the deep feature extractor's output embedding distances of the sketch and the image have been used to retrieve the top k closest images from the image database of the unseen classes.

# Results on unseen(zero-shot) classes

The below table presents a few qualitative results of our model, which was not trained on any of these classes. The table presents the top 5 results from left to right.

| Query Sketch    | Retrieved Image 1 | Retrieved Image 2 | Retrieved Image 3 | Retrieved Image 4 | Retrieved Image 5 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
|![](docs/s1.png)|![](docs/i11.png)|![](docs/i12.png)|![](docs/i13.png)|![](docs/i14.png)|![](docs/i15.png)|
|![](docs/s2.png)|![](docs/i21.png)|![](docs/i22.png)|![](docs/i23.png)|![](docs/i24.png)|![](docs/i25.png)|

# Architecture Overview

# Instructions
<details>
<summary>
Installation
</summary>

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


