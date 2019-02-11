## WORK IN PROGRESS

Implementation of : ZERO-RESOURCE MULTILINGUAL MODEL TRANSFER- LEARNING WHAT TO SHARE



### Ressources
Dataset Amazon review data
    * Disponible ici https://webis.de/data/webis-cls-10.html
    * Issu du papier : http://anthology.aclweb.org/P/P10/P10-1114.pdf

Embeddings :
    * Get FastText emb : https://github.com/Kyubyong/wordvectors
    * Official FastText emb https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
    * Then apply vecmap https://github.com/artetxem/vecmap
    * Multi lingual word embeddings : https://github.com/Babylonpartners/fastText_multilingual

Comments : https://openreview.net/forum?id=SyxHKjAcYX



## "à la main"
Dans MUSE le txt alignment_matrices/ja.txt => alignment_matrices/jp.txt
japonais => jp (initialement jp pour MUSE)

## Naive model
A partir des embeddings alignés => naive_model.py

## Shared Features (SF) model => pf_model.py

## Final model => SF model + PF model => final_model.py


