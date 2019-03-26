### 1. 为什么机器学习策略
### 2. 如何使用这本书
### 3. 先决条件
### 4. 规模驱动的机器学习
### 5. 验证和测试集
dev set： 验证集 \
验证集应该来自应用场景中的数据，而不应该是来自“与训练集同分布”

### 6. dev和test 应该来自同分布
### 7. dev／test 需要多大
一般来说
For example, if classifier A has an accuracy of 90.0% and classifier B has an accuracy of 90.1%, then a dev set of 100 examples would not be able to detect this 0.1% difference. 也就是说验证集应该尽可能大，如此才能反应问题，上面的例子来说，验证集至少要1000。
### 8. 设置一个single-number 评估值来不断优化
要设置一个目标！
### 9. 优化和满足度量 satisficing metrics
### 10. 用dev 和metrics 来加速迭代
### 11. 何时改变dev／test以及度量
### 12. 设置development 和测试集
----
## 错误分析
### 13. 快速建立第一个系统，然后迭代
Even if the basic system is far from the “best” system you can build, it is valuable to examine how the basic system functions: you will quickly find clues that show you the most promising directions in which to invest your time. 
### 14. 错误分析：看dev 集中的例子来评估idea
Before investing a month on this task, I recommend that you first **estimate how much it will actually improve the system’s accuracy**. Then you can more rationally decide if this is worth the month of development time, or if you’re better off using that time on other tasks.
In detail, here’s what you can do:
1. Gather a sample of 100 dev set examples that your system  misclassified . I.e., examples that your system made an error on.
2. Look at these examples manually, and count what fraction of them are dog images.
The process of looking at misclassified examples is called  error analysis . In this example, if you find that only 5% of the misclassified images are dogs, then no matter how much you improve your algorithm’s performance on dog images, you won’t get rid of more than 5% of your errors. In other words, **5% is a “ceiling” (meaning maximum possible amount) for how much the proposed project could help**. 

我导师就每次就是这样的。
### 15. 错误分析时，并行评估多个方法
如下图
(./15.png)
### 16. 清除错标的数据
错标导致的分类错误也是一类错误，当这个部分比较大时，就要考虑清除或者校正，校正时也需要double check那些分类正确的样本中是否同样含有错标数据，同时在训练集和测试集中要使用相同的策略校正。
### 17. 如果dev集比较大，那么分成两个子集
- Let​’​s continue our example above, in which the algorithm is misclassifying 1,000 out of 5,000 dev set examples. Suppose we want to manually examine about 100 errors for error analysis (10% of the errors). You should randomly select 10% of the dev set and place that into what we’ll call an **​Eyeball dev set** to remind ourselves that we are looking at it with our eyes. 另外的90%（4500） 则称为** Blackbox dev set** 
- If you see the performance on the Eyeball dev set improving much more rapidly than the performance on the Blackbox dev set, you have **overfit** the Eyeball dev set. In this case, you might need to discard it and find a new Eyeball dev set by **moving more examples from the Blackbox dev set into the Eyeball dev set** or by acquiring new labeled data.
### 18. Eyeball 和Blackbox 应该多大
对于人类也表现很好的分类情景：
- 至少要100个mistake cases
- The lower your classifier’s error rate, the larger your Eyeball dev set needs to be in order to get a large enough set of errors to analyze.
- 如果dev set本身就很小，那首先保证Eyeball，也就是对出现的所有errors 都进行分析。当然这样的风险就是可能会过拟合。
### 19. 基本的错误分析总结 
----
## 偏差和方差
### 20. 偏差和方差
Suppose as above that your algorithm has 16% error (84% accuracy) on the dev set. We break the 16% error into two components:
- First,the algorithm’s error rate on the training set.In this example,it is 15%. We think of this informally as the algorithm’s ​bias​.
- Second,how much worse the algorithm does on the dev(or test) set than the training set. In this example, it does 1% worse on the dev set than the training set. We think of this informally as the algorithm’s ​variance​.

误差=偏差+方差，因此减小偏差意味着提升模型，减小方差意味着改善泛化。
Some changes to a learning algorithm can address the first component of error—​bias​—and improve its performance on the training set. Some changes address the second component—​variance​—and help it generalize better from the training set to the dev/test sets.
### 21. 偏差和方差的例子
- 偏差即训练时的error rate，方差即测试集上的error rate - 训练时的error rate。
- 过拟合：低偏差，高方差（1%， 10%）
- 欠拟合：高偏差，低方差（15%，1%）
- 如果都很低，那是一个好的分类器；如果都很高（15%，16%） 那么针对过拟合/欠拟合的技术就不好使了。
### 22. 与optimal error rate比较
optimal error rate （Bayes error rate or Bayes rate）：可以理解为理想误差，大多数情况下就是human performance。比如训练误差是15%，如果人类的分类误差接近0%（例如对猫狗分类）那么这就是一个高偏差/欠拟合的 情况，如果人类的分类误差是14%（例如语音识别），那么这个分类器在拟合方面进步空间就不大了。因此进一步将bias分成两类：
- Avoidable bias：error we attribute to the learning algorithm’s bias，即training error - optimal error rate，**如果这是一个负数，那就是过拟合了**。
- Optimal error rate (“unavoidablebias”)​:即 optimal error rate

### 23. 处理 偏差和方差
- 高偏差（不特意说明即为Avoidable bias）：increase the size of your model（例如增加神经网络的层数）
- 高方差（增加训练数据量）
- Increasing the model size generally reduces bias, but it might also increase variance and the risk of overfitting. However, this overfitting problem usually arises only when you are not using regularization. If you include a well-designed regularization method, then you can usually safely increase the size of the model without increasing overfitting. （正则化）
### 24. 偏差方差 tradeoff
### 25. 降低偏差的技术
- Increase the model size：如果过拟合，那就加正则化
- Modify input features based on insights from error analysis：基于错误分析增加特征，这对bias和 viarance都有好处，如果过拟合，就加正则
- Reduce or eliminate regularization：可以降低bias，但是很可能会增大方差
- Modify model architecture：对方差和偏差都有好处
- Add more training data：理论上说对biaos是没有帮助的。
### 26. 在训练集上进行错误分析
**Your algorithm must perform well on the training set before you can expect it to perform well on the dev/test sets.** \
This can be useful if your algorithm has high bias—i.e., if it is not fitting the training set well. 错误分析的方法同以上类似
### 27. 降低方差的方法
- Add more training data
- **Add regularization**
- Add early stopping：Early stopping behaves a lot like regularization methods, and some authors call it a regularization technique.
- Feature selection to decrease number/type of input features：当训练集比较小的时候，做特征选择还是可能会有帮助的。
- Modify input features based on insights from error analysis：同上
- Modify model architecture
## 学习曲线
### 28. 诊断偏差和方差：学习曲线
训练集增大可以减小测试集上的误差，但是会增大训练集上的误差。
### 29. 画出训练误差
### 30. 理解学习曲线： 高偏差
如图 (./30-%E9%AB%98%E5%81%8F%E5%B7%AE.png)
### 31. 理解学习曲线： 其他case
### 32. 画出学习曲线
----
## Comparing to human-level performance
### 33. 为什么要比较human-level performance
### 34. 如何定义human-level performance
- human labelers
- desired optimal error rate
- Error analysis can draw on human intuition.
### 35. 超越human-level
机器超越human-level 是可能的，但只要在某些case上人类能取得正确结果而机器错误，那么机器就还有提升的空间。
## Training and testing on different distributions
### 36. 什么时候应该在不同分布上训练和测试
- Choose dev and test sets to reflect data you expect to get in the future and want to do well on. 验证集和测试集一定要与将要应用的场景同分布，而训练集可以加入外部数据源。
- But in the era of big data, we now have access to huge training sets, such as cat internet images. Even if the training set comes from a different distribution than the dev/test set, we still want to use it for learning since it can provide a lot of information.
### 37. 如何决定是否用全部的数据
假设额外的数据源有20000的互联网数据，与应用场景同分布的数据源10000个（5000训练，5000dev）那么什么时候使用这2w的数据源：
- 如果model是传统的model，hand-designed computer vision features, followed by a simple linear classifier，那么就不能使用这额外的数据源；
- 如果是现在的大数据（神经网络）那可以将这部分数据加入到模型，但是在带来收益的同时也可能expend some of its capacity to learn about properties that are specific to internet images (such as higher resolution, different distributions of how the images are framed, etc.) ，所以如果high resolution是应用场景普遍存在的问题，那么可能会hurt the alghthrom performace.
- 除了考虑模型外，还要考虑这额外的数据集是否相关，例如猫狗分类中，这2w的数据集中不包含猫狗的负样本是一些文本扫描图片，那么就需要把这部分无关图片剔除，免得浪费计算资源。
### 38. 如何决定是否包含不一致数据
inconsistent：数据源会影响结果，比如要预测纽约的房价，是否可以用华盛顿的房价数据，这就是不一致的；相反，猫狗分类的互联网照片与用户上传的照片就是一致的。
### 39. weighting data
例如用户上传的照片有5000张，从互联网获取的额外数据有200000张，大概是40:1，那么这种比例很悬殊的情况下可以加权重来平衡两者的重要性，比如增大用户上传的权重或者减少互联网数据的权重。
### 40. 从训练集到dev set的泛化
泛化不好分两种情况：
- 高方差：测试集与训练集同分布，在测试集上的error rate 高于 训练集上的；
- data mismatch： 在与训练集同分布的测试集error rate较小，但是在不同分布的测试集上error rate较大（例如在训练时加入了很多额外数据）。
### 41. 识别方差／偏差和data mismatch error
回到猫分类的例子，如果在训练时加入了很多互联网图片，测试时误差高于训练误差，那么此时应该做一个诊断：是data mismatch 还是高方差。
- 把训练集分出一部分作为training dev set，这部分不用做训练，用来得到“同分布测试集的error rate”；
- 如果上面的误差与训练误差相近，那么此时是data mismatch；如果上面的误差与测试集相近，那么是高方差 ，可以根据之前chapter的方法进行改善。
### 42. addressing （解决）data mismatch
- 看错误分类数据--> to understand the significant differences between the training and the dev set 
- Try to find more training data that better matches the dev set examples (可能不太行)
### 43. 人工数据合成
例如加噪声。\
很有挑战，因为很容易过拟合，可能要花费一点时间来找到恰当的合成方法。
## Debugging inference algorithms
### 44. 最优化验证测试？the optimization verification test
### 45. 从上面的泛化
### 46. 强化学习例子
## End-to-end deep learning
### 47. 端到端学习
### 48. 更多端到端学习的例子
### 49. 端到端学习的pros and cons
### 50. 选择管道成分：数据可获得？
### 51. 选择管道成分： 任务简化？
### 52. 直接学习 rich outputs
## Error analysis by parts
### 53. 错误分析 by parts
例如猫分类器分为两个parts：detector and classifier. 
### 54. 把错误归因到某个part
例如一个region包含半只猫，在对这个case进行归因时不确定是detector or classifier， 那么可以给classifier喂一个人工标注的标准region，看其是否分类正确，如果此时分类正确，那这个case应该归因于detector， 否则应该归因于classifier.
### 55. 错误分布的一般case
如果有多个A->B->C...多个parts，也是先假设（人工处理）A的输出时perfect，然后测试B/C的输出是否正确，依次来将case归因。
### 56. 错误分析与human-level 
At its most basic, error analysis by parts tells us what component(s) performance is (are) worth the greatest effort to improve. \
归因时为了找到最有效的提升方向。

**Many error analysis processes work best when we are trying to automate something humans can do and can thus benchmark against human-level performance.**  
因为人可以做到很好，因此you have more powerful error analysis tools.

例如自动驾驶规划路径：视觉需要检测周边的车/行人，这两个parts 都是有human-level可以参考的。
### 57. 辨认出一个有缺陷的ML pipeline
例如上面的自动驾驶规划路径，如果检测汽车/行人都接近human-level，但仍然很差，那可能是这个pipeline的原因。 maybe the inputs do not contain enough information, etc.
## Conclusion
### 58.

