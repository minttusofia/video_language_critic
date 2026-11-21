# Video-Language Critic (VLC): Transferable Reward Functions for Language-Conditioned Robotics
Official implementation of:
<p align="center"><b>Video-Language Critic: Transferable Reward Functions for Language-Conditioned Robotics</b><br>
Minttu Alakuijala, Reginald McLean, Isaac Woungang, Nariman Farsad, Samuel Kaski, Pekka Marttinen, Kai Yuan<br>
<a href=https://arxiv.org/abs/2405.19988>[Paper]</a><br>

## Setup

```
git clone https://github.com/minttusofia/video_language_critic.git
cd video_language_critic

conda env create -f vlc.yml
conda activate vlc

pip install -e .
```
Make sure PyTorch 2.1 is installed with CUDA support. See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for the installation corresponing to your version of CUDA. E.g. `conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

## Usage

### Meta-World training
To train VLC on Meta-World videos in data/metaworld/mw50_videos (to reproduce experiments in Section 4.1):
```
torchrun main.py --num_thread_reader 6 --epochs 20 --batch_size 64 --n_display 20 --data_path data/metaworld/mw50_videos --features_path data/metaworld/mw50_videos --output_dir experiments/mw50_training --seed 1 --lr 1e-4 --batch_size_val 64 --datatype mw --loss_type sequence_ranking_loss --ranking_loss_weight 33 --coef_lr 1e-3 --use_failures_as_negatives_only --slice_framepos 3 --test_slice_framepos 2 --augment_images --sim_header tightTransf --pretrained_clip_name ViT-B/32 --other_eval_metrics strict_auc,tv_MeanR,vt_MedianR,vt_R1,tv_R1,tv_R10,tv_R5,labeled_auc,vt_loss --do_train --n_ckpts_to_keep -1
```
To train on MW40 videos (Section 4.2), use `--data_path data/metaworld/mw40_split` and change `--output_dir` to differentiate the output directories.


### Open X-Embodiment training

To train VLC on Open X-Embodiment videos, first download the dataset using instructions in https://github.com/google-deepmind/open_x_embodiment.

We used the dataset metadata to download only splits that include language annotations. This list of splits can also be found in `LANG_DATASETS` in `./video_language_critic/dataloaders/dataset_info/openx.py`. Our dataloader expects the data in the original TFrecord format, organised into subdirectories by split and version, e.g. `OPENX_DATA_DIRECTORY/bc_z_0.1.0/0.1.0/bc_z-train.tfrecord-00000-of-01024` to `bc_z-train.tfrecord-01023-of-01024`.

To train VLC on Open X videos contained in OPENX_DATA_DIRECTORY (Section 4.3):
```
torchrun main.py --num_thread_reader 6 --epochs 15 --batch_size 64 --n_display 200 --data_path OPENX_DATA_DIRECTORY --output_dir experiments/openx_training --test_data_path data/vlmbench/test_picks --test_features_path data/vlmbench/test_picks --test_set_name vlmbench/test_picks --test_datatype vlm --seed 1 --lr 1e-4 --batch_size_val 64 --datatype openx --loss_type cross_entropy --coef_lr 1e-3 --use_failures_as_negatives_only --slice_framepos 3 --test_slice_framepos 2 --augment_images --sim_header tightTransf --pretrained_clip_name ViT-B/32 --other_eval_metrics strict_auc,tv_MeanR,vt_MedianR,vt_R1,tv_R1,tv_R10,tv_R5,labeled_auc,vt_loss --do_train --n_ckpts_to_keep -1
```

## Trained models

The trained models can be downloaded [here](https://aaltofi-my.sharepoint.com/:f:/g/personal/minttu_alakuijala_aalto_fi/EvDb_h6Dum5Joh703sCA9JIBJ16VPFz79_lJkTowv53VEg?e=znFaAY).

## RL training

To train RL policies with the reward models, please see [https://github.com/reginald-mclean/VLC_RL](https://github.com/reginald-mclean/VLC_RL).


## Citation

If you found this implementation or the trained models useful, please cite our work as
```bibtex
@article{alakuijala2025videolanguage,
    title={Video-Language Critic: Transferable Reward Functions for Language-Conditioned Robotics},
    author={Minttu Alakuijala and Reginald McLean and Isaac Woungang and Nariman Farsad and Samuel Kaski and Pekka Marttinen and Kai Yuan},
    journal={Transactions on Machine Learning Research},
    year={2025},
}
```


## Acknowledgments
Our code is based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).
