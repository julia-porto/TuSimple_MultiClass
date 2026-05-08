# TuSimple_MultiClass
Multiclass labelling (continuous, dashed, unmarked) of TuSimple lane segmentation benchmark. Dataset available at [Kaggle](https://www.kaggle.com/datasets/julporto/tusimple-multiclass/data).
For the original dataset, see https://github.com/TuSimple/tusimple-benchmark/blob/master/README.md

This work has been accepted for presentation at the [WCTR 2026 Conference](https://www.wctr2026.fr/). Full conference paper will be uploaded as soon as available.

This repository contains three main information:
1. The script used for labeling (MultiClassAssing.py)
2. The scripts used for training a segmentation model on that data
3. The metadata for the best model trained so far. The actual model can be made available upon request.

All changes made in the original TuSimple dataset are described in CHANGES.md

To run `MultiClassAssing.py`, you will need to have downloaded the original TuSimple dataset and saved in a folder named "tusimple_processed_split" (or change the name in the python file).

For the segmentation model, a LinkNet model with dual-branch BYOL/LinkNet fine-tuning was trained. The code used to train is stored in scripts, where:
- load_metadata.py is the script to load the multiclass metadata
- multiclasses.py contains all functions and classes
- run_code.py actually runs the training, tests and saves it locally.
