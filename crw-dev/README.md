## CRW-Dev Dataset: README

This dataset is a supplement to the CRW Dataset [1]. It was constructed by randomly 
selecting 550 pairs of words from the Rare Words dataset [2] which do not occur in the 
CRW Dataset and occur at least 128 times in the Westbury Wikipedia corpus (WWC) [3].

The contents of this directory are as follows:

- CRW-550.txt: This file contains the 550 pairs of words. Each line contains three 
  entries that are separated by tabs: the two words (of which the second 
  one is the rare word) and their similarity.
   
- rarevocab.txt: This file contains all 414 rare words used in the 550 pairs.

- context/<rareword>.txt: This file contains at least 128 sentences from the WWC 
  in which <rareword> occurs. Each line contains exactly one sentence. A such file 
  exists for each of the 414 rare words listed in rarevocab.txt.
 
## References 

[1] Khodak, M.; Saunshi, N.; Liang, Y.; Ma, T.; Stewart, B.; and Arora, S. 2018. 
    A la carte embedding: Cheap but effective induction of semantic feature vectors. 
    In Proceedings of the 56th Annual Meeting of the Association for Computational 
    Linguistics (Volume 1: Long Papers), 12-22. Association for Computational 
    Linguistics.

[2] Luong, T.; Socher, R.; and Manning, C. 2013. Better word representations with 
    recursive neural networks for morphology. In Proceedings of the Seventeenth 
    Conference on Computational Natural Language Learning, 104-113.

[3] Shaoul, C., and Westbury, C. 2010. The westbury lab wikipedia corpus.
