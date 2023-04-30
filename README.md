# BACH-GCN

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


