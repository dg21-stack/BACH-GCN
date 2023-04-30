# BACH-GCN

Final Project for CSCI3397 Biomedical Image Analysis

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
Change model name where stated above, do the same for line 108. (WARNING: doing so also requires changing models down the line for generating graph embeddings.

