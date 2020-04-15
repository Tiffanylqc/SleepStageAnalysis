# Automated Sleep Stage Classification
A sequence to sequence learning model for sleep stage and fragmentation analysis with ready to use server API. 
[Github repo](https://github.com/Tiffanylqc/SleepStageAnalysis) 
## Requirements
* Python 2.7
* tensorflow/tensorflow-gpu
* numpy
* scipy
* scikit-learn
* matplotlib
* imbalanced-learn(0.4.3)
* pandas
* mne

## Data Preparation
We use [the Physionet Sleep-EDF datasets](https://physionet.org/physiobank/database/sleep-edfx/) for training our model. 
* Use the below script to download SC subjects from the Sleep_EDF (2013) dataset

```
cd data_2013
chmod +x download_physionet.sh
./download_physionet.sh
```
* Use the below script to preprocess the signals from specified folder, channel

```
python prepare_upload_signal.py --data_dir data_2013 --output_dir data_2013/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'
```

## Train Model
You can modify args settings in seq2seq_sleep_sleep-EDF.py at your discretion. 
* Use the below script to train a model with the 20-fold cross-validation using Fpz-Cz channel of the Sleep_EDF (2013) dataset:
```
python seq2seq_sleep_sleep_EDF.py --data_dir data_2013/eeg_fpz_cz --output_dir output_2013 --n_folds 20
```

## Generate Classification for a Single Signal
Use the below script to first preprocess a signal and then classify different stages for it with the selected model saved in folder `/seq2seq-model`.
```
python prepare_upload_signal.py --data_dir eeg_data --output_dir eeg_data/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'
python predict_upload_signal.py --filename ST7022J0.npz
```
## Sleep Fragmentation Analysis and Scoring
Helper functions for scoring, sleep fragmentation analysis locate in script `eeg_frag_info_combine.py`. These functions will be used in our API in `deploy.py`

## Deploy as a server for Web APP
Use the below script to start a local Flask server 
```
python deploy.py
```
### Supported API
* `POST /api/analysis`
    * Body: `file=[sleep signal file]` 
    * Header: `ContentType=multipart/form-data`
    * Response:
        - message: HTTP status message
        - sleep_score, sleep_msg: sleep quality score and the message to display on clicking the button of score
        - wtr, wtr_msg: weighted transition rate and the message to display on clicking the button of wtr
        - total_sleep_time: total actual sleep time to be displayed on FE
        - first_section_info: the information about the first section
        - total_section_num: total number of sections
        ```
        {  
            "message": ...,
            "sleep_score": ...,
            "sleep_msg": ...,
            "wtr": ...,
            "wtr_msg": ...,
            "total_sleep_time": ...,
            "total_section_num": ...,
            "first_section_info": {
                "section_id": ...,
                "stage_name": ...,
                "description": ..., 
                "signal": {
                        "x": ...,
                        "y": ...,
                }
            }
        }
        ```
* `GET /api/section`
    * Param: `sec_id=[int from 0 to total_section_num-1]`
    * Response:
        ```
        {
            "section": {
                "section_id": ...,
                "stage_name": ...,
                "description": ..., 
                "signal": {
                        "x": ...,
                        "y": ...,
                }
            }
        }
        ```
## Miscellaneous
* `dataloader.py dhedfreader.py` are helper libraries for reading edf signal files

* `Procfile requirements.txt runtime.txt` are config files for hosting the server on [Heroku](https://www.heroku.com)
## References
[github:akaraspt](https://github.com/akaraspt/deepsleepnet)  
[github:MousaviSajad](https://github.com/MousaviSajad/SleepEEGNet)