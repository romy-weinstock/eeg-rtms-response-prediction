# EEG-Based Prediction of rTMS Treatment Response in Depression

   Investigating whether pre-treatment EEG connectivity and oscillator-synchrony features can predict rTMS response in MDD, framed as a predictive-enrichment problem for CNS drug development.

   ## Data

This project uses the TDBRAIN dataset (van Dijk et al., 2022), provided by the Brainclinics Foundation under a Data Use Agreement. Data is licensed under CC BY 4.0; accompanying preprocessing code is licensed under MIT.

Citation: van Dijk, H., van Wingen, G., Denys, D. et al. The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN) database. Sci Data 9, 333 (2022). https://doi.org/10.1038/s41597-022-01409-z

Data source: https://brainclinics.com/resources/tdbrain-dataset

Raw data is not included in this repository (see .gitignore) and is subject to the terms of the TDBRAIN Data Use Agreement, including restrictions on redistribution and re-identification of subjects.

## Update regarding preprocessing pipeline

The original plan was to use the TDBRAIN authors' published preprocessing pipeline directly. On inspection, this code does not run on the current dataset release: it expects CSV files with a fixed 33-channel layout and a legacy filename convention, neither of which match the BDF/BIDS-formatted files provided in the current TDBRAIN V3.1 dataset. This is a compatibility gap between the published code and the dataset's more recent format update, not a limitation of the methodology itself.

To address this, preprocessing for this project reimplements the documented methodology from van Dijk et al. (2022) natively in MNE-Python, rather than using the original code directly. Specifically, EOG artifact correction uses the regression-based method published by Gratton et al. (1983), matching the authors' documented approach, the ICA-based artifact removal explored in the MNE fundamentals notebook (`00_mne_fundamentals_tutorial.ipynb`) was tool-learning, not the method used in the final pipeline.
