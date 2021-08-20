# TrUSt
PyTorch code of our CIKM paper  
[TrUMAn: Trope Understanding in Movies and Animations](https://arxiv.org/abs/2108.04542)
## TrUMAn Dataset
Website: [https://www.cmlab.csie.ntu.edu.tw/project/trope/](https://www.cmlab.csie.ntu.edu.tw/project/trope/)  
Data download: [TrUMAn](https://www.cmlab.csie.ntu.edu.tw/project/trope/#data)
Video examples: [Trope video](https://www.cmlab.csie.ntu.edu.tw/project/trope/#explore)

We present a novel dataset TrUMAn (Trope Understanding in Movies and Animations) which includes (1) 2423 videos with audio, (2) 132 tropes along with human-annotated categories. We also include human-annotated video descriptions in our dataset to compare the domain gap between raw visual and audio signals and human-written text. We classify tropes into 8 categories by their properties. As categories are not orthogonal, some tropes belong to 2 or more categories.
## Model overview
A multi-stream multi-task model, each stream process different contextual inputs. And performs two tasks (1) Storytelling (2) Trope Understanding
![image](https://github.com/pwshen/TrUSt/blob/main/imgs/model.png)
## Requirement
* python 3.6  
* PyTorch 1.5
* numpy
* h5py
* tdqm
* pickle
## Usage
