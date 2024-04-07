# Response to Reviewer EKht

We thank you for your careful review and address your concerns below. 

To better resolve your concerns, we have conducted extra experiments by including new models (Swin Transformer V2 [1]) and new datasets (ImageNet-1K [2]). Unless otherwise stated, all results in this rebuttal are reported on the 8 models from different model families (ResNet-[18, 34, 50, 101, 152] and SwinV2-[tiny(t), small(s), base(b)]), as suggested in Q6. Similar to our setup in Sec 5.1, the updated normalized costs for all the models are as follows.

| Models | Normalized Cost | 
| :--------: | :--------: |
| ResNet-18 | 0.15 |
| ResNet-34 | 0.22 |
| ResNet-50 | 0.29 |
| ResNet-101 | 0.52 |
| ResNet-152 | 0.76 |
| SwinV2-t | 0.53 |
| SwinV2-s | 0.98 |
| SwinV2-b | 1 |

**Q1: The authors have an extensive theoretical justification for their accuracy estimator. However, the theory is quite straightforward and superfluous. Lemma 4.5 says that the proposed estimator is unbiased, which is not surprising since it relies on the assumption that as the sample size increases, the distance to the nearest neighbor approaches 0. However, on high-dimensional data one needs an enormous sample size to make this in any way meaningful as Min_x’ dist(x,x’) is almost certainly very large.**

A1: Thank you for this comment. We address this concern as follows.
1. We politely disagree with the claim saying our theory is “straightforward and superfluous”. Perhaps our accessible presentation gives the impression that the theoretical justification is straightforward. Below, we point out the non-trivial challenges that needed to be solved to develop the theoretical justification. 
2. Since Lemma 4.5 was quoted in this comment, we will use it as an example to demonstrate the challenges. Firstly, we would like to kindly clarify that Lemma 4.5 does not rely on an “assumption” but a well-derived result from Lemma 4.4, that as the sample size increases, the distance to the nearest neighbor approaches 0. The proof of Lemma 4.4 is not trivial since it relies on the existence of a Lipschitz continuous oracle classifier (an ideal classifier outputting ground truth labels), which has been shown to be not universally possible for classification problems [3, 4, 5]. We address this challenge by utilizing the well-separation property of the image classification problem (see Def 4.1), which has been well observed in the literature [6], based on which we prove that such an oracle classifier exists in our formulation (see Lemma 4.3) and pave the way for developing principled accuracy estimators with statistical guarantees.
3. We further conducted experiments to investigate the distance to the nearest neighbor (min_x’ dist(x,x’)) as sample size (s) increases. We report results with different feature extractors (ResNet-18 and SwinV2-t) as well as different metrics (L_inf, L1, and L2) on the test split of TinyImageNet dataset, as shown below.

![alt text](https://github.com/OCCAM2024/OCCAM/blob/main/min_dist_res18.png?raw=true)
![alt text](https://github.com/OCCAM2024/OCCAM/blob/main/min_dist_swinv2t.png?raw=true)

It can be clearly seen that the distance to the sampled nearest neighbour quickly approaches 0 as sample size increases. This could be attributable to the fact that we are sampling from real images. With properly pre-trained feature extractors, the possible image embeddings could be restricted to a subspace rather than pervade the whole high-dimensional space, which can significantly reduce the required number of samples and give us meaningfully small distances to the sampled nearest neighbours. 


**Q2: Following the theoretical exposition, the formulation of the problem into an integer linear programming problem is straightforward.**

A2: Thank you for this comment. We would like to stress that the ILP formulation is a natural choice given the deep connection between our problem and the classic multiple choice knapsack problem (MCKP) [7]. Recall that our optimal model portfolio problem seeks the optimal assignment of ML classifiers that maximizes the overall accuracy under given cost budgets. After having the cost and accuracy estimation for each ML classifier (see Sec 4.1), our problem can be reduced to the problem of “selecting for each query image, one item (i.e., ML classifier) from a collection (the set of all classifiers) so as to maximize the total value (accuracy) while adhering to a predefined weight limit (cost budget)”. This is an MCKP problem at its core and the ILP formulation is the natural choice. The ILP formulation allows us to leverage the recent progress in generic ILP solver development, which helps deliver quality results both efficiently and effectively, as demonstrated in Sec 5.1. 

**Q3: The article only considers image classification, and experiments are performed using three small data sets (CIFAR10, CIFAR100, Tiny ImageNet) using only ResNets. Moreover, the baselines are quite weak, and the simple "single best" baseline already works quite well in the experiments.**

A3: Thank you for this comment. We address your concerns below.
1. As to the task, image classification by itself is a fundamental task in computer vision that has great real-world implications. In addition, the image classification problem has been observed to enjoy the well-separation property [6] which is critical to derive statistical guarantees (see Sec 4.1). As discussed in Sec 6, it is intriguing to explore the possibility of extending our approach to other tasks (e.g., sentiment analysis in NLP) and we will investigate it in our future work.
2. As to the models and datasets, we have conducted extra experiments with more models (Swin Transformer V2) and datasets (ImageNet-1K), as mentioned at the beginning of our rebuttal. On ImageNet-1K, we randomly sample 10,000 query images from the validation split for evaluation purposes and use the rest to compute the accuracy estimator. We use a sample size s=4,000 on ImageNet-1K and perform bootstrap sampling given the scarcity of labelled evaluation data per category. All other settings use the default choices as described in Sec 5.1. We report the cost reduction and accuracy drop trade-off below.

|   TinyImageNet-200   |             | Accuracy Drop |     (%)    |          |
|:--------------------:|:-----------:|:-------------:|:----------:|:--------:|
| Cost  Reduction  (%) | Single Best |      Rand     | Frugal-MCT |    OMP   |
|          10          |     4.98    |      7.26     |    0.57    | **0.28** |
|          20          |     4.98    |      7.26     |    1.21    | **0.87** |
|          40          |     4.01    |      7.26     |    3.42    | **2.46** |

|      ImageNet-1K     |             | Accuracy Drop |     (%)    |          |
|:--------------------:|:-----------:|:-------------:|:----------:|:--------:|
| Cost  Reduction  (%) | Single Best |      Rand     | Frugal-MCT |    OMP   |
|          10          |     5.37    |      5.99     |   **0.9**  |   0.95   |
|          20          |     5.37    |      5.99     |    1.38    | **1.14** |
|          40          |     2.53    |      5.99     |    2.62    | **1.97** |

Clearly, OCCAM outperforms all baselines across a majority of experimental settings, similar to our observation in Sec 5.2. Specifically, with 40% cost reduction, OCCAM outperforms FrugalMCT by achieving 0.96% and 0.65% higher accuracy on TinyImageNet and ImageNet separately, which demonstrates the effectiveness of OCCAM with stringent budget limits.

3. As for the baselines, we consider three – Rand (random accuracy estimation), FrugalMCT (ICML 2022) [8], and Single Best. Of these, Rand was used as a baseline in previous work [9], and following them we use it as a baseline to demonstrate the effectiveness of our accuracy estimator. FrugalMCT is the most recent work related to our problem and is SOTA, which serves as the strongest baseline. Single Best chooses the single best model for all queries given a cost budget. Though it works well in high budget cases, we would like to point out that its performance drastically decays as the budgets go down (see Figures 3, 5-6). This matters, as in practical scenarios, it is critical to achieve good performance with limited budgets, making Single Best not a good fit.


**Q4: Since the proposed method relies on nearest neighbor search using extracted ResNet features, I am not convinced that the results would necessarily be similar on other model families. The method's usability is also limited due to the fact that it needs to first extract these features for the query. Such a feature might not be available when the method is used with available image classification APIs, and a (potentially significant) portion of the budget is already used by the extraction step.**

A4: Thank you for this valuable comment. We address your concerns below.
1. As described in A3, we have conducted extra experiments by including new model families (Swin Transformer V2). Our approach is observed to outperform all baselines which demonstrates its effectiveness and generalizability.
2. We use logits from the last layer of ML classifiers as the features (see Sec 5.1), which are part of the standard outputs from common commercial APIs (e.g., Google Cloud Vision API [10], Microsoft Azure Vision API [11]). Also, the rich family of publicly accessible pre-trained ML models ensures the accessibility of feature extractors and the usability of our approach.
3. In our approach, the cheapest ML classifier is used to extract features by default, which is supposed to incur a small cost (e.g., the normalized cost of ResNet-18 w.r.t. SwinV2-b is only 0.15). Moreover, the computation results in feature extraction are reused to generate predictions which saves the classification costs at later stages and alleviates the concern of “potentially significant” extraction costs. 

**Q5: The authors should shorten the theoretical section as it provides no meaningful insight, and instead focus on more extensive experiments if possible.**

A5. Thank you for the comment. We politely disagree with the claim saying our theory “provides no meaningful insight”. We would like to point out the important implications of our theoretical findings.
1. (Lemma 4.3) A Lipschitz continuous classifier of perfect accuracy (oracle classifier) is possible for the image classification task, which is non-trivial because such an oracle classifier has been found to be not universally tractable for general classification problems [3, 4, 5]. 
2. (Lemma 4.4) Given a query image, the likelihood that a Lipschitz continuous classifier makes the right prediction is also a Lipschitz continuous function, which is the mathematical justification for the well-accepted intuition “a robust image classifier should **likely** have similar performance on similar inputs” and is non-trivial without the development of Lemma 4.3, as discussed in our A1.
3. (Lemmas 4.5 & 4.6) We provide a complete proof showing that our estimator is asymptotically unbiased and low-variance. As pointed out in Sec 2, a majority of previous work typically trains ML models to predict the accuracy, which not only requires sophisticated configuration but also lacks performance guarantees that are critical in real-world scenarios. To our best knowledge, we are the first to open up the black box by developing a white-box accuracy estimator for ML classifiers with statistical guarantees. We believe this is an interesting and inspiring contribution and sincerely hope that you would reconsider your rating.  

We will include the new experiment results and improve the presentation of the theoretical section in our camera-ready version to better bring out the significance of the contribution.

**Q6: In particular, I would recommend that the authors extend their experiments to cover more models, including potentially a setting where there are models from multiply different model families. In addition, there is not really anything in the method that prevents it from being used for other tasks than image classification and such tasks could also be considered.**

A6: Thank you for this comment. As described in our A3, we have conducted extra experiments on models from different families (ResNets and Swin Transformers) and we observed that our approach outperforms all baselines across a majority of experimental settings. As to the image classification task, as mentioned in our response in A3, the image classification task has been found to satisfy the well-separation property [6] which is important for developing statistical guarantees.We will explore the well-separation property of other tasks in the future so that this work can be extended to other tasks, with statistical guarantees.   

**Q7: The authors should include additional ablation studies on how accurate the proposed estimator is, and how much the method is affected by the choice of the model used for extracting the features used to perform the nearest neighbor search.**

A7: Thank you for this comment. We have investigated the estimation error (difference between real classifier accuracy and our estimator results) for different ML classifiers as well as the performance changes of OCCAM, using different feature extractors (ResNet-18, ResNet-50, and SwinV2-t). 

For brevity, on TinyImageNet, we report the estimation error in the accuracy of all 8 classifiers (ResNet-[18, 34, 50, 101, 152], and SwinV2-[t, s, b]), under L_inf metric. The patterns are similar with other metrics and feature extractors.

![alt text](https://github.com/OCCAM2024/OCCAM/blob/main/est_error_res18.png?raw=true)
![alt text](https://github.com/OCCAM2024/OCCAM/blob/main/est_error_res50.png?raw=true)
![alt text](https://github.com/OCCAM2024/OCCAM/blob/main/est_error_swinv2t.png?raw=true)

It is clear that the estimation error of our accuracy estimator continues to decrease as the sample size (s) increases, which demonstrates the effectiveness of our accuracy estimator.

We further report the performance of OCCAM with different feature extractors (ResNet-18, ResNet-50, and SwinV2-t), on TinyImageNet. As in Sec 5.1, the costs incurred by feature extraction are “deducted from the user budget before we compute the optimal model portfolio”. Results are summarized below.

|       ResNet-18      |             | Accuracy Drop |     (%)    |          |
|:--------------------:|:-----------:|:-------------:|:----------:|:--------:|
| Cost  Reduction  (%) | Single Best |      Rand     | Frugal-MCT |    OMP   |
|          10          |     4.98    |      7.26     |    0.57    | **0.28** |
|          20          |     4.98    |      7.26     |    1.21    | **0.87** |
|          40          |     4.01    |      7.26     |    3.42    | **2.46** |

|       ResNet-50      |             | Accuracy Drop |     (%)    |          |
|:--------------------:|:-----------:|:-------------:|:----------:|:--------:|
| Cost  Reduction  (%) | Single Best |      Rand     | Frugal-MCT |    OMP   |
|          10          |     4.98    |      7.26     |    0.57    | **0.42** |
|          20          |     4.98    |      7.26     |    1.21    | **0.68** |
|          40          |     4.01    |      7.26     |    3.42    | **2.45** |

|       SwinV2-t       |             | Accuracy Drop |     (%)    |          |
|:--------------------:|:-----------:|:-------------:|:----------:|:--------:|
| Cost  Reduction  (%) | Single Best |      Rand     | Frugal-MCT |    OMP   |
|          10          |     4.98    |      7.26     |    0.57    | **0.15** |
|          20          |     4.98    |      7.26     |    1.21    | **0.46** |
|          40          |     4.01    |      7.26     |    3.42    | **2.83** |


It can be seen that OCCAM outperforms all baselines on all experimental settings, which demonstrates the effectiveness and generalizability of OCCAM with different feature extractors.

**Q8: Another setting that could potentially be more interesting is to consider inference latency instead of dollar cost, which has a similar formulation but also requires one to consider the latency of the allocation itself.**

A8: Thank you for this valuable comment. We address your concerns below.
1. In practice, dollar costs usually highly correlate with inference latency on GPUs, while allocation is performed on CPUs which are typically cheap and the incurred costs could be negligible. In our work, the dollar costs are computed by multiplying inference latency with standard unit prices ($/hour), as discussed in Sec 5.1. The results under inference latency settings can be straightforward mapped from our dollar cost results.
2. Considering allocation latency itself in the allocation optimization is an interesting problem. A naive baseline is to have a fixed budget for allocation itself and develop a progressive allocation strategy which can stop and output the best-so-far result whenever it runs out of allocation budgets. We will investigate other possibilities in our future work.

Thank you for your time and consideration. We sincerely hope that you would consider increasing your rating in light of our responses. 


References:
1. Liu, Ze, et al. "Swin transformer v2: Scaling up capacity and resolution." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
2. Russakovsky, Olga, et al. "Imagenet large scale visual recognition challenge." International journal of computer vision 115 (2015): 211-252.
3. Fawzi, Alhussein, Hamza Fawzi, and Omar Fawzi. "Adversarial vulnerability for any classifier." Advances in neural information processing systems 31 (2018).
4. Tsipras, Dimitris, et al. "Robustness may be at odds with accuracy." arXiv preprint arXiv:1805.12152 (2018).
5. Zhang, Hongyang, et al. "Theoretically principled trade-off between robustness and accuracy." International conference on machine learning. PMLR, 2019.
6. Yang, Yao-Yuan, et al. "A closer look at accuracy vs. robustness." Advances in neural information processing systems 33 (2020): 8588-8601.
7. Kellerer, Hans, et al. "The multiple-choice knapsack problem." Knapsack Problems (2004): 317-347.
8. Chen, Lingjiao, Matei Zaharia, and James Zou. "Efficient online ml api selection for multi-label classification tasks." International conference on machine learning. PMLR, 2022.
9. Ding, Dujian, et al. "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing." The Twelfth International Conference on Learning Representations. 2023.
10. https://cloud.google.com/vision?hl=en
11. https://azure.microsoft.com/en-us/products/ai-services/ai-vision/
