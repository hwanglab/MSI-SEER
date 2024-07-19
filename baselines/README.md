# Comparative Analysis for MSI-H prediction

In our experimental evaluation, we juxtapose our DGP-based prediction approach with CNN-based deep learning models, including ResNet, ShuffleNet, and EfficientNet, following the same training methodologies as outlined in [Laleh et al., Medical Image Analysis, 2022](https://www.sciencedirect.com/science/article/pii/S1361841522001219). Our comparative analysis takes inspiration from and extends the foundation laid by their research.

## CNN-Based Deep Learning Models

We leveraged the implementations available at the provided [source](https://github.com/KatherLab/HIA), employing three distinct CNN architectures with pre-trained weights on the ImageNet dataset.

All reference models undergo an 8-epoch training process utilizing the Adam optimization algorithm, with a batch size of 128, a learning rate set to 1e-4, and an L2 weight decay of 1e-5. Throughout the training phase, we consistently store model iterations that exhibit the lowest validation loss, ensuring optimal performance.

### [MSIDETECT](https://jnkather.github.io/msidetect/?fireglass_rsn=true#fireglass_params&tabid=aa57b94d82485517&start_with_session_counter=2&application_server_address=mc9.prod.fire.glass)

Additionally, we employed [models](https://zenodo.org/record/5151502#.ZFHjeXbMKUl) trained for MSI-H prediction derived from previous research, performing independent training and evaluation on our dataset to further validate our approach.

## Run baseline
```
CUDA_VISIBLE_DEVICES=0 python train_patch_level.py --k 0 --model_name efficientnet --exp train_TCGA_CRC_Kather_3fold --splits data_info/TCGA_CRC_Kather_3fold.csv
```
- `--seed`: random seed for reproducible experiment
- `--k`: fold number
- `--model_name`: model architecture, choices=['resnet', 'efficientnet', 'shufflenet']
- `--freeze`: freeze ratio of layers
- `--train_dir`: train data csv file directory
- `--test_dir`: test data csv file directory
- `--result_dir`: results root directory
- `--exp`: experiment name for experiment folder directory')
- `--splits`: data split csv file directory for k-fold cross validation
- `--epochs`: number of epochs to train 
- `--patience`: number of patience for early stopping 
- `--lr`: learning rate 
- `--weight_decay`: weight decay 
- `--batch_size`: batch size 

## References
- [Laleh et al., Medical image analysis 2022](https://www.sciencedirect.com/science/article/pii/S1361841522001219)
- [Kather et al., Nature Medicine 2019](https://www.nature.com/articles/s41591-019-0462-y)
