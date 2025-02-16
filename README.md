# explainable_MNG_spike_detection

An experimental project for spike detection in Microneurography recordings [4] with an explainable neural network solution.

## Installation

The original model is written in PyTorch, and the pipeline uses pandas, numpy, and matplotlib. For the model analysis, iNNvestigate is used, for which a TensorFlow implementation is also needed. 

```sh
conda env create -f environment.yml
```

## Data

The dataset used in this project is private. However, its structure and sample visualizations can be found in the `raw_data_loading_plotting.ipynb` notebook.


## Structure:

| **File/Folder**                      | **Description**                                                                                              |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `external/`                          | PyTorch and TensorFlow VPNet and WHVPNet implementations ([1][2])|
| `trained_models/`                    | The best trained model|
| `MicroneurographyDataloader.py`      | Custom dataloader for loading, sliding-window preparation, data augmentation, labeling, and train-validation-test splitting |
| `XAIProject.py`                      | Custom class for interpretability analysis|
| `metrics.py`                         | Common and merged metrics with proximity and latency filtering. Boxplot visualization for model certainty analysis.|
| `model_analysis.ipynb`               | Decision analysis using the iNNvestigate library and relevance score plotting|
| `raw_data_loading_plotting.ipynb`    | Data exploration and visualization|
| `spike_classification.py`            | Training script for classification model|
| `visualize_output.ipynb`             | Predicted sliding windows transformed back into the original continuous signal space and visualized as a colormap based on predictions. Proximity and latency filtering can also be applied here.|
| `training_loading_visualizing.ipynb` | Combined pipeline for training and visualization|

## Reference:

- [1] T. Dózsa, C. Böck, J. Meier, and P. Kovács, "Weighted Hermite Variable Projection Networks for Classifying Visually Evoked Potentials," in IEEE Transactions on Neural Networks and Learning Systems, 2024, doi: 10.1109/TNNLS.2024.3475271.
- [2] Kovács P, Bognár G, Huber C, Huemer M. VPNET: Variable Projection Networks. Int J Neural Syst. 2022 Jan;32(1):2150054. doi: 10.1142/S0129065721500544. Epub 2021 Oct 13. PMID: 34651549.
- [3] Alber, M., Lapuschkin, S., Seegerer, P., Hägele, M., Schütt, K. T., Montavon, G., Samek, W., Müller, K.-R., Dähne, S., & Kindermans, P.-J. (2019). iNNvestigate Neural Networks! Journal of Machine Learning Research, 20(93), 1–8. Retrieved from [JMLR](http://jmlr.org/papers/v20/18-540.html)
- [4] Kutafina, E., Troglio, A., De Col, R., Röhrig, R., Rossmanith, P., & Namer, B. (2022). Decoding Neuropathic Pain: Can We Predict Fluctuations of Propagation Speed in Stimulated Peripheral Nerve? Frontiers in Computational Neuroscience, 16, 899584. https://doi.org/10.3389/fncom.2022.899584

