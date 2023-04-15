# H2CGL
H2CGL: Modeling Dynamics of Citation Network for Impact Prediction

## Requirement

* pytorch >= 1.10.2
* numpy >= 1.13.3
* sklearn
* python 3.9
* transformers
* dgl == 0.91

## Usage
Original dataset:
* PMC(pubmed): https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
* DBLPv13(dblp): https://www.aminer.org/citation

Download the dealt dataset from https://pan.baidu.com/s/1BkcIUiyMbdn9FR3hLdws1w&pwd=kr7h <br>

You can get the H2CGL model here https://drive.google.com/file/d/1nJEgQdTenTssbZfjywe6poiOQKYvJpBo/view?usp=sharing

### Training
```sh
# H2CGL
python main.py --phase H2CGL --data_source h_pubmed/h_dblp --cl_type label_aug_hard_negative --aug_type cg --encoder_type 'CGIN+RGAT' --n_layers 4 --hn 2 --hn_method co_cite
```

### Testing

```sh
# H2CGL
python main.py --phase test_results --model H2CGL --data_source h_pubmed/h_dblp --cl_type label_aug_hard_negative --aug_type cg --encoder_type 'CGIN+RGAT' --n_layers 4 --hn 2 --hn_method co_cite
```