# BACH-GCN:

Final Project for CSCI3397 Biomedical Image Analysis

For this project I proposed a novel solution the 2019 BACH Challenge via the usage of pretrained hact-net -- a neural network model which combines both cell and tissue based features into the same graph embeddings -- cell and tissue graph NN. After generating graph embeddings the embeddings were then trained via Light GBM, SVM and Logistic Regression.

To further broaden my scope of research, I also used two different architectures to tweak the graph representation rendering -- VGG16 and InceptionV3. These two architectures were decided upon because the solution I was using for reference also utilized ResNet, VGG16 and InceptionV3 for generation of features before plugging into LightGBM.


# Requirements:

installation via requirements.txt

# Generating Feature Extractions:

After cloning git reposity do the following to generate cell and tissue graphs as well as assignment matrices: 
```
cd BACH-GCN\core
```

```
 python generate_hact_graphs.py --image_path <PATH>\<TO>\<IMGS> --save_path <PATH>\<TO>\<FEATURES>\hack-net-data
```
The features generated will be based off of a resnet34 unsupervised feature extraction process. In order to change models, go to 'generate_hact_graphs.py' and to line 83:

```python
  self.nuclei_feature_extractor = DeepFeatureExtractor(
            architecture='<CHANGE THIS>',
            patch_size=72,
            resize_size=224
        )
```
Change model name where stated above, do the same for line 108. (WARNING: doing so also requires changing models down the line for generating graph embeddings.) 

# Generating Graph Embeddings:

Once appropriate feature extraction is fulfilled and graph embeddings are to be generated the following must be done: 

```
python train.py --cg_path \<PATH TO CELL GRAPH>\cell_graphs\ --tg_path <PATH TO TISSUE GRAPH>\tissue_graphs\ --assign_mat_path <PATH TO ASSIGNMENT MATRICES\assignment_matrices\  --config_fpath .\config\bracs_<MODEL OF CHOICE>_7_classes_pna.yml
```
Replace model of choice with either: 'hact', 'tggnn', or 'cggnn'

For each graph embeddings generated, change save text at the bottom of ```train.py``` accordingly:

```
    x = np.array(x)
    print(x)
    np.savetxt('<CHANGE NAME FOR X VALUES>.txt',x)
    np.savetxt('<CHANGE NAME FOR Y VALUES>.txt',np.array(y))
````
 
 
# Training via Light GBM:

Open ```trainer.py``` and run through intstructions

# VGG16 Results:

```
hact-multi log loss 0.83
hact-binary log loss 0.59
cell multiclass gbm: 1.1284330448956537
cell normal binary gbm: 0.5995631511529192
tissue multiclass gbm: 1.1287772738965263
tissue normal binary gbm: 0.640050119228732

0.7096774193548387 h
0.5161290322580645 c
0.5483870967741935 t

0.7096774193548387 h
0.7096774193548387 c
0.7096774193548387 t

 ---------------- HACT RESULTS ---------------- 
multiclass support vector classification
              precision    recall  f1-score   support

         0.0       0.50      0.67      0.57         6
         1.0       0.71      0.62      0.67         8
         2.0       0.88      0.54      0.67        13
         3.0       0.38      0.75      0.50         4

    accuracy                           0.61        31
   macro avg       0.62      0.65      0.60        31
weighted avg       0.70      0.61      0.63        31

binary normal support vector classification
              precision    recall  f1-score   support

           0       0.67      0.71      0.69        17
           1       0.62      0.57      0.59        14

    accuracy                           0.65        31
   macro avg       0.64      0.64      0.64        31
weighted avg       0.64      0.65      0.64        31

 ---------------- HACT RESULTS ---------------- 
multiclass logistic regression
              precision    recall  f1-score   support

         0.0       0.62      0.56      0.59         9
         1.0       0.57      0.44      0.50         9
         2.0       0.62      0.50      0.56        10
         3.0       0.38      1.00      0.55         3

    accuracy                           0.55        31
   macro avg       0.55      0.62      0.55        31
weighted avg       0.59      0.55      0.55        31

binary normal logistic regression
              precision    recall  f1-score   support

           0       0.67      0.71      0.69        17
           1       0.62      0.57      0.59        14

    accuracy                           0.65        31
   macro avg       0.64      0.64      0.64        31
weighted avg       0.64      0.65      0.64        31


 ---------------- CELL RESULTS ---------------- 
multiclass support vector classification
              precision    recall  f1-score   support

         0.0       0.75      0.43      0.55        14
         1.0       0.12      0.17      0.14         6
         2.0       0.40      0.40      0.40         5
         3.0       0.60      1.00      0.75         6

    accuracy                           0.48        31
   macro avg       0.47      0.50      0.46        31
weighted avg       0.54      0.48      0.48        31

binary normal support vector classification
              precision    recall  f1-score   support

           0       0.65      0.69      0.67        16
           1       0.64      0.60      0.62        15

    accuracy                           0.65        31
   macro avg       0.64      0.64      0.64        31
weighted avg       0.65      0.65      0.64        31


---------------- CELL RESULTS ---------------- 
multiclass logistic regression
              precision    recall  f1-score   support

         0.0       0.75      0.43      0.55        14
         1.0       0.12      0.20      0.15         5
         2.0       0.40      0.40      0.40         5
         3.0       0.60      0.86      0.71         7

    accuracy                           0.48        31
   macro avg       0.47      0.47      0.45        31
weighted avg       0.56      0.48      0.50        31

binary normal logistic regression
              precision    recall  f1-score   support

           0       0.65      0.69      0.67        16
           1       0.64      0.60      0.62        15

    accuracy                           0.65        31
   macro avg       0.64      0.64      0.64        31
weighted avg       0.65      0.65      0.64        31


 ---------------- TISSUE RESULTS ---------------- 
multiclass support vector classification
              precision    recall  f1-score   support

         0.0       0.38      0.60      0.46         5
         1.0       0.50      0.50      0.50        12
         2.0       0.29      0.22      0.25         9
         3.0       0.75      0.60      0.67         5

    accuracy                           0.45        31
   macro avg       0.48      0.48      0.47        31
weighted avg       0.46      0.45      0.45        31

binary normal support vector classification
              precision    recall  f1-score   support

           0       0.62      0.62      0.62        16
           1       0.60      0.60      0.60        15

    accuracy                           0.61        31
   macro avg       0.61      0.61      0.61        31
weighted avg       0.61      0.61      0.61        31

 ---------------- TISSUE RESULTS ---------------- 
multiclass logistic regression
              precision    recall  f1-score   support

         0.0       0.38      0.60      0.46         5
         1.0       0.50      0.50      0.50        12
         2.0       0.29      0.29      0.29         7
         3.0       1.00      0.57      0.73         7

    accuracy                           0.48        31
   macro avg       0.54      0.49      0.49        31
weighted avg       0.54      0.48      0.50        31

binary normal logistic regression
              precision    recall  f1-score   support

           0       0.56      0.60      0.58        15
           1       0.60      0.56      0.58        16

    accuracy                           0.58        31
   macro avg       0.58      0.58      0.58        31
weighted avg       0.58      0.58      0.58        31



```
