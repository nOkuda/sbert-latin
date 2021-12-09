# Latin SBERT

Code used to train and experiment with Latin SBERT.

## Installation

Using a Python virtual environment is recommended.

A CUDA-capable GPU is required to run the current code.

  1. Install pytorch with the correct CUDA version for your machine (<https://pytorch.org/get-started/locally/>). Reported results used 1.9.
  2. Install other Python dependencies
     ```
     pip3 install -r requirements.txt
     ```
  3. Install CLTK Latin tokenizer models
     ```
     python3 -c "from cltk.corpus.utils.importer import CorpusImporter; corpus_importer = CorpusImporter('latin'); corpus_importer.import_corpus('latin_models_cltk')"
     ```
  4. Download Latin BERT model and subtokenizer as well as .tess files
     ```
     ./download.sh
     ```

Note that it is necessary for `pytorch` to be installed before `transformers`, which is a library listed in `requirements.txt`.

Version numbers have been purposely pinned (particularly `cltk` and `nltk`) in order for the code to work properly.
It is likely that someone with more time could investigate how to update this code to work with newer versions of the libraries.

## Running Code

### Training the model

To train a model, run `train.py`.
This will create an `output` directory, where Latin SBERT training information is stored.
The first subdirectory will be the random seed used during training of Latin SBERT.
Within that subdirectory will be further subdirectories: `best_model`, `checkpoint`, `dev_eval`, `final_model`, `test_eval`, and `train_eval`.

The `*_eval` directories contain model performance on each of the three data sets: the training, development, and test sets.
Each has a `parallels.txt`, many `predictions_*.txt`, and a `record.txt` file.
The `parallels.txt` file contains all of the parallels in the data set.
The text before the first tab is a sentence from the _Aeneid_, the text between the first and second tabs is a sentence from the _Bellum Civile_, and the number following the second tab is the label given to the sentence pair.
The text after the newline following the number is a sentence for the next entry in the data set.
The `predictions_*.txt` files list two columns of numbers.
The first column lists the values the model predicted, and the second column lists the true label.
Each line in a `predictions_*.txt` file corresponds to the ordered entries in `parallels.txt`.
The `record.txt` file is tab separated file that collects model performance measurements.
The first row is a header indicating what the numbers in the associated column measure.

The `best_model`, `checkpoint`, and `final_model` directories contain savepoints for the model.
The model saved in `best_model` is the model with the highest development set Spearman rank correlation coefficient.
The model saved in `checkpoint` is the last model trained for an epoch number ending in 0.
The model saved in `final_model` is the model after the final epoch.

### Verifying pre-training performance

To check the numbers on a model without any training, run `check_start.py`.
Because the random number seed is set the same as `train.py`, the training, development, and test sets should be the same in both scripts.

### Plotting `output` information

Running `plot.py` will generate training plots: one tracking Spearman rank correlation, and one tracking mean squared error loss.
They will be called `eval_plot.svg` and `loss_plot.svg`, respectively.

### Running the experiment

To run the 5-fold cross-validation experiment, run `cross_val.py`.
This will create an `output_cross_val` directory where experiment information is stored.
The `fold` directory has one subdirectory for each fold of the cross validation.
Within each subdirectory are a `checkpoint`, `final_model`, `test_eval`, and `train_eval` directory.
These hold information analogous to the directories under the `output` directory (described above) with the same name.

To train and evaluate models on the Latin SBERT output, run `more_cross_val.py` after running `cross_val.py`.
This will generate a `metrics.txt` file in the `output_cross_val` directory
This `metrics.txt` file contains information about evaluation measurements on the various models and the tasks they were trained on.
Also, various `.svg` files displaying confusion matrices will be generated in the directory.
The `.svg` files are named according to the task name and the model trained on the confusion matrix they display.
All `.svg` files generated either start with `learnplot` or `reliabilityplot`.
Files starting with `learnplot` are colored across the rows, highlighting model recall; these are the ones presented in the dissertation chapter.
Files starting with `reliabilityplot` are colored across the columns, highlighting model precision.

### Other Variations

In addition to the original Latin SBERT training and experiment code,
there are a few variations with results that didn't go anywhere.

The `mini_*.py` scripts printed out performance data at the end of every batch instead of at the end of every epoch.

The `balanced_*.py` scripts used weighted resampling to keep the classes a little more even during training.
Weightings were set so that each class had equal probability of appearing in each batch.
Although the `balanced_mini_*.py` series of scripts suggested that weighted resampling was helping,
the `balanced_*cross_val.py` scripts indicated slightly worse performance than the original experiments of this repo.

## Acknowledgments

The most important inspirations for this code were Latin BERT (<https://github.com/dbamman/latin-bert>) and SBERT (<https://github.com/UKPLab/sentence-transformers>).
Adaptations from CLTK (<https://github.com/cltk/cltk>), NLTK (<https://github.com/nltk/nltk>), and tensor2tensor (<https://github.com/tensorflow/tensor2tensor>) were also crucial in making the code work.
