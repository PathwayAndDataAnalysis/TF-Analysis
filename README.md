# Rank Based Transcription Factor Analysis


## Introduction

The fundamental structure of this algorithm can be described as follows: initially, it randomly selects the combined number of up-regulated and down-regulated targets from the maximum possible rank. For instance, consider a symbol with 15 up-regulated and 3 down-regulated targets, and a maximum possible rank of 414. In this situation, the algorithm selects 18 random integers from the range of 0 to 414 and generates a secondary list representing the reverse ranking of the 18 randomly chosen numbers. Subsequently, it identifies the cumulative up-regulated and down-regulated targets from the 18 randomly chosen ranks, considering both the positive and negative perspectives. The lesser value among these two perspectives is deemed the final rank-sum for the given symbol.




## Requirements

- Python 3.8 or higher
- numpy 1.19.2 or higher
- pandas 1.1.3 or higher



## Installation

```bash
git clone https://github.com/PathwayAndDataAnalysis/TF-Analysis.git
cd TF-Analysis/main

python old_tf_analysis.py -cp simulated/simulated_priors.tsv -sc dim_data/simulated_data.tsv -iters 100000 -o simulated/result.tsv -sim 1

```



## License

GNU LESSER GENERAL PUBLIC LICENSE


