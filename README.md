# Revisiting Counterfactual Regression through the Lens of Gromov-Wasserstein Information Bottleneck

## 1 Abstract

As a promising individualized treatment effect (ITE) estimation method, counterfactual regression (CFR) maps individuals' covariates to a latent space and predicts their counterfactual outcomes. 
However, the selection bias between control and treatment groups often imbalances the two groups' latent distributions and negatively impacts this method's performance.
In this study, we revisit counterfactual regression through the lens of information bottleneck and propose a novel learning paradigm called Gromov-Wasserstein information bottleneck (GWIB).
In this paradigm, we learn CFR by maximizing the mutual information between covariates' latent representations and outcomes while penalizing the kernelized mutual information between the latent representations and the covariates.
We demonstrate that the upper bound of the penalty term can be implemented as a new regularizer consisting of $i)$ the fused Gromov-Wasserstein distance between the latent representations of different groups and $ii)$ the gap between the transport cost generated by the model and the cross-group Gromov-Wasserstein distance between the latent representations and the covariates. 
GWIB effectively learns the CFR model through alternating optimization, suppressing selection bias while avoiding trivial latent distributions. 
Experiments on ITE estimation tasks show that GWIB consistently outperforms state-of-the-art CFR methods.

## 2 Quick Start

Choose a model (e.g., GWIB) to run with the following command.

```
    python main.py --lr 0.01 --batchSize 64 --beta 0.5 --lambda 0.1
```


## 3 Hyper-parameters search range

We tune hyper-parameters according to the following table.

| Hyper-parameter | Explain                                     | Range                                 |
| --------------- | ------------------------------------------- | ------------------------------------- |
| lr              | learning rate                               | \{0.00001, 0.0001, 0.001, 0.01, 0.1\} |
| bs              | batch size of each mini-batch               | \{16, 32, 64, 128\}                   |
| dim_backbone    | the dimensions of representation            | \{32, 64\}                            |
| dim_task        | the dimensions of prediction head           | \{32, 64\}                            |
| beta            | weight of fused Gromov-Wasserstein distance | \{0.1, 0.3, 0.5, 0.7, 0.9\}           |
| lambda          | weight proposed OT-based regularization     | \{0.0001, 0.001, 0.01, 0.1, 1\}       |

## 4. Paper
arXiv link: http://arxiv.org/abs/2405.15505

Citation:
> @misc{yang2024revisiting,
>      title={Revisiting Counterfactual Regression through the Lens of Gromov-Wasserstein Information Bottleneck},
>      author={Hao Yang and Zexu Sun and Hongteng Xu and Xu Chen},
>      year={2024},
>      eprint={2405.15505},
>      archivePrefix={arXiv},
>      primaryClass={cs.LG}
>      }
