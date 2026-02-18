# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
# DeepSeekMath: 突破开源语言模型数学推理能力的极限


Zhihong Shao ${}^{1,2 *  \dagger  }$ ,Peiyi Wang ${}^{1,3 *  \dagger  }$ ,Qihao Zhu ${}^{1,3 *  \dagger  }$ ,Runxin Xu ${}^{1}$ ,Junxiao Song ${}^{1}$ Xiao ${\mathrm{{Bi}}}^{1}$ ,Haowei Zhang ${}^{1}$ ,Mingchuan Zhang ${}^{1}$ ,Y.K. Li ${}^{1}$ ,Y. Wu ${}^{1}$ ,Daya Guo ${}^{1 * }$
邵执弘 ${}^{1,2 *  \dagger  }$ ,王佩宜 ${}^{1,3 *  \dagger  }$ ,朱其豪 ${}^{1,3 *  \dagger  }$ ,徐润欣 ${}^{1}$ ,宋俊晓 ${}^{1}$ 肖 ${\mathrm{{Bi}}}^{1}$ ,张浩威 ${}^{1}$ ,张明川 ${}^{1}$ ,Y.K. Li ${}^{1}$ ,Y. Wu ${}^{1}$ ,郭大雅 ${}^{1 * }$


${}^{1}$ DeepSeek-AI, ${}^{2}$ Tsinghua University, ${}^{3}$ Peking University
${}^{1}$ DeepSeek-AI, ${}^{2}$ 清华大学, ${}^{3}$ 北京大学


\{zhihongshao,wangpeiji,zhugh,guoday\}@deepseek.com https://github.com/deepseek-ai/DeepSeek-Math
\{zhihongshao,wangpeiji,zhugh,guoday\}@deepseek.com https://github.com/deepseek-ai/DeepSeek-Math


## Abstract
## 摘要


Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pretraining DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.
由于数学推理具有复杂且结构化的特性，这对语言模型构成了重大挑战。在本文中，我们推出了 DeepSeekMath 7B，该模型在 DeepSeek-Coder-Base-v1.5 7B 的基础上，利用来自 Common Crawl 的 120B 数学相关标记（Token）以及自然语言和代码数据进行了持续预训练。DeepSeekMath 7B 在不依赖外部工具包和投票技术的情况下，在竞赛级 MATH 基准测试中取得了 51.7% 的优异成绩，接近 Gemini-Ultra 和 GPT-4 的性能水平。在 64 个样本的自一致性评测下，DeepSeekMath 7B 在 MATH 上达到了 60.9%。DeepSeekMath 的数学推理能力归功于两个关键因素：首先，我们通过精心设计的选择流程，挖掘了公开网络数据的巨大潜力；其次，我们引入了组相对策略优化（GRPO），这是近端策略优化（PPO）的一种变体，它在增强数学推理能力的同时，优化了 PPO 的显存占用。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_17_11_83097c.jpg"/>



Figure 1 | Top1 accuracy of open-source models on the competition-level MATH benchmark (Hendrycks et al. 2021) without the use of external toolkits and voting techniques.
图 1 | 开源模型在竞赛级 MATH 基准测试（Hendrycks et al. 2021）中不使用外部工具包和投票技术的 Top1 准确率。


---



* Core contributors.
* 核心贡献者。


† Work done during internship at DeepSeek-AI.
† 在 DeepSeek-AI 实习期间完成的工作。


---



## 1. Introduction
## 1. 引言


Large language models (LLM) have revolutionized the approach to mathematical reasoning in artificial intelligence, spurring significant advancements in both the quantitative reasoning benchmark (Hendrycks et al., 2021) and the geometry reasoning benchmark (Trinh et al., 2024). Moreover, these models have proven instrumental in assisting humans in solving complex mathematical problems (Tao, 2023). However, cutting-edge models such as GPT-4 (OpenAI 2023) and Gemini-Ultra (Anil et al. 2023) are not publicly available, and the currently accessible open-source models considerably trail behind in performance.
大语言模型（LLM）彻底改变了人工智能处理数学推理的方法，推动了定量推理基准（Hendrycks et al., 2021）和几何推理基准（Trinh et al., 2024）的显著进步。此外，这些模型已被证明在辅助人类解决复杂数学问题方面发挥了重要作用（Tao, 2023）。然而，GPT-4（OpenAI 2023）和 Gemini-Ultra（Anil et al. 2023）等尖端模型并未开源，而目前可获取的开源模型在性能上仍大幅落后。


In this study, we introduce DeepSeekMath, a domain-specific language model that significantly outperforms the mathematical capabilities of open-source models and approaches the performance level of GPT-4 on academic benchmarks. To achieve this, we create the DeepSeek-Math Corpus, a large-scale high-quality pre-training corpus comprising 120B math tokens. This dataset is extracted from the Common Crawl (CC) using a fastText-based classifier (Joulin et al., 2016). In the initial iteration, the classifier is trained using instances from OpenWebMath (Paster et al. 2023) as positive examples, while incorporating a diverse selection of other web pages to serve as negative examples. Subsequently, we employ the classifier to mine additional positive instances from the CC, which are further refined through human annotation. The classifier is then updated with this enhanced dataset to improve its performance. The evaluation results indicate that the large-scale corpus is of high quality, as our base model DeepSeekMath-Base 7B achieves 64.2% on GSM8K (Cobbe et al., 2021) and 36.2% on the competition-level MATH dataset (Hendrycks et al., 2021), outperforming Minerva 540B (Lewkowycz et al., 2022a). In addition, the DeepSeekMath Corpus is multilingual, so we notice an improvement in Chinese mathematical benchmarks (Wei et al. 2023; Zhong et al. 2023). We believe that our experience in mathematical data processing is a starting point for the research community, and there is significant room for improvement in the future.
在本研究中，我们推出了 DeepSeekMath，这是一款特定领域的语言模型，其数学能力显著超越了现有的开源模型，并在学术基准上接近 GPT-4 的水平。为此，我们构建了 DeepSeek-Math Corpus，这是一个包含 120B 数学 Token 的大规模高质量预训练语料库。该数据集使用基于 fastText 的分类器（Joulin et al., 2016）从 Common Crawl (CC) 中提取。在初始迭代中，分类器使用来自 OpenWebMath（Paster et al. 2023）的样本作为正例，并结合多样化的其他网页作为负例进行训练。随后，我们利用该分类器从 CC 中挖掘更多正例，并通过人工标注进一步精炼。分类器随后使用该增强数据集进行更新以提升性能。评估结果表明，该大规模语料库质量极高：我们的基础模型 DeepSeekMath-Base 7B 在 GSM8K（Cobbe et al., 2021）上达到 64.2%，在竞赛级 MATH 数据集（Hendrycks et al., 2021）上达到 36.2%，表现优于 Minerva 540B（Lewkowycz et al., 2022a）。此外，由于 DeepSeekMath 语料库是多语言的，我们观察到其在中文数学基准测试（Wei et al. 2023; Zhong et al. 2023）上也有所提升。我们相信，我们在数学数据处理方面的经验可以作为研究界的起点，未来仍有巨大的提升空间。


DeepSeekMath-Base is initialized with DeepSeek-Coder-Base-v1.5 7B (Guo et al., 2024), as we notice that starting from a code training model is a better choice compared to a general LLM. Furthermore, we observe the math training also improves model capability on MMLU (Hendrycks et al. 2020) and BBH benchmarks (Suzgun et al. 2022), indicating it does not only enhance the model's mathematical abilities but also amplifies general reasoning capabilities.
DeepSeekMath-Base 基于 DeepSeek-Coder-Base-v1.5 7B (Guo et al., 2024) 初始化，因为我们注意到，与通用大语言模型相比，从代码训练模型开始是更好的选择。此外，我们观察到数学训练还提升了模型在 MMLU (Hendrycks et al. 2020) 和 BBH 基准测试 (Suzgun et al. 2022) 上的能力，这表明它不仅增强了模型的数学能力，还放大了通用推理能力。


After pre-training, we apply mathematical instruction tuning to DeepSeekMath-Base with chain-of-thought (Wei et al. 2022), program-of-thought (Chen et al. 2022) Gao et al. 2023), and tool-integrated reasoning (Gou et al., 2023) data. The resulting model DeepSeekMath-Instruct 7B beats all 7B counterparts and is comparable with 70B open-source instruction-tuned models.
预训练后，我们利用思维链 (Wei et al. 2022)、程序思维 (Chen et al. 2022; Gao et al. 2023) 和工具集成推理 (Gou et al., 2023) 数据对 DeepSeekMath-Base 进行数学指令微调。所得模型 DeepSeekMath-Instruct 7B 击败了所有 7B 级别的对手，且可与 70B 开源指令微调模型媲美。


Furthermore, we introduce the Group Relative Policy Optimization (GRPO), a variant reinforcement learning (RL) algorithm of Proximal Policy Optimization (PPO) (Schulman et al., 2017). GRPO foregoes the critic model, instead estimating the baseline from group scores, significantly reducing training resources. By solely using a subset of English instruction tuning data, GRPO obtains a substantial improvement over the strong DeepSeekMath-Instruct, including both in-domain (GSM8K: 82.9% → 88.2%, MATH: 46.8% → 51.7%) and out-of-domain mathematical tasks (e.g., CMATH: 84.6% $\rightarrow$ 88.8%) during the reinforcement learning phase. We also provide a unified paradigm to understand different methods, such as Rejection Sampling Fine-Tuning (RFT) (Yuan et al., 2023a), Direct Preference Optimization (DPO) (Rafailov et al., 2023), PPO and GRPO. Based on such a unified paradigm, we find that all these methods are conceptualized as either direct or simplified RL techniques. We also conduct extensive experiments, e.g., online v.s. offline training, outcome v.s. process supervision, single-turn v.s. iterative RL and so on, to deeply investigate the essential elements of this paradigm. At last, we explain why our RL boosts the performance of instruction-tuned models, and further summarize potential directions to achieve more effective RL based on this unified paradigm.
此外，我们引入了组相对策略优化 (GRPO)，这是一种近端策略优化 (PPO) (Schulman et al., 2017) 的变体强化学习 (RL) 算法。GRPO 放弃了判别器模型，改为从组得分中估计基线，从而显著减少了训练资源。仅通过使用一部分英语指令微调数据，GRPO 在强化学习阶段就使强大的 DeepSeekMath-Instruct 获得了实质性提升，包括领域内 (GSM8K: 82.9% → 88.2%, MATH: 46.8% → 51.7%) 和跨领域数学任务 (例如 CMATH: 84.6% $\rightarrow$ 88.8%)。我们还提供了一个统一范式来理解不同的方法，如拒绝采样微调 (RFT) (Yuan et al., 2023a)、直接偏好优化 (DPO) (Rafailov et al., 2023)、PPO 和 GRPO。基于该统一范式，我们发现所有这些方法在概念上都可以被视为直接或简化的强化学习技术。我们还进行了广泛的实验（例如在线与离线训练、结果与过程监督、单轮与迭代强化学习等），以深入探究该范式的核心要素。最后，我们解释了为什么我们的强化学习能够提升指令微调模型的性能，并进一步总结了基于该统一范式实现更有效强化学习的潜在方向。


### 1.1. Contributions
### 1.1. 贡献


Our contribution includes scalable math pre-training, along with the exploration and analysis of reinforcement learning.
我们的贡献包括可扩展的数学预训练，以及对强化学习的探索与分析。


## Math Pre-Training at Scale
## 大规模数学预训练


- Our research provides compelling evidence that the publicly accessible Common Crawl data contains valuable information for mathematical purposes. By implementing a meticulously designed data selection pipeline, we successfully construct the DeepSeekMath Corpus, a high-quality dataset of 120B tokens from web pages filtered for mathematical content, which is almost 7 times the size of the math web pages used by Minerva (Lewkowycz et al., 2022a) and 9 times the size of the recently released OpenWebMath (Paster et al. 2023).
- 我们的研究提供了有力证据，证明公开获取的 Common Crawl 数据包含对数学用途有价值的信息。通过实施精心设计的数据选择流程，我们成功构建了 DeepSeekMath 语料库，这是一个包含 120B token 的高质量数据集，源自过滤数学内容的网页，其规模几乎是 Minerva (Lewkowycz et al., 2022a) 所用数学网页的 7 倍，是近期发布的 OpenWebMath (Paster et al. 2023) 的 9 倍。


- Our pre-trained base model DeepSeekMath-Base 7B achieves comparable performance with Minerva 540B (Lewkowycz et al., 2022a), indicating the number of parameters is not the only key factor in mathematical reasoning capability. A smaller model pre-trained on high-quality data could achieve strong performance as well.
- 我们的预训练基座模型 DeepSeekMath-Base 7B 达到了与 Minerva 540B (Lewkowycz et al., 2022a) 相当的性能，这表明参数量并非数学推理能力的唯一关键因素。在高质量数据上预训练的小型模型同样可以获得强劲的表现。


- We share our findings from math training experiments. Code training prior to math training improves models' ability to solve mathematical problems both with and without tool use. This offers a partial answer to the long-standing question: does code training improve reasoning abilities? We believe it does, at least for mathematical reasoning.
- 我们分享了数学训练实验的发现。在数学训练之前进行代码训练，可以提高模型在有或无工具辅助下解决数学问题的能力。这为长期存在的问题提供了一个部分答案：代码训练是否能提高推理能力？我们认为确实如此，至少对于数学推理是这样的。


- Although training on arXiv papers is common, especially in many math-related papers, it brings no notable improvements on all mathematical benchmarks adopted in this paper.
- 尽管在 arXiv 论文上进行训练很常见，尤其是在许多数学相关论文中，但在本文采用的所有数学基准测试上，它并未带来显著的提升。


## Exploration and Analysis of Reinforcement Learning
## 强化学习的探索与分析


- We introduce Group Relative Policy Optimization (GRPO), an efficient and effective reinforcement learning algorithm. GRPO foregoes the critic model, instead estimating the baseline from group scores, significantly reducing training resources compared to Proximal Policy Optimization (PPO).
- 我们引入了组相对策略优化 (GRPO)，这是一种高效且有效的强化学习算法。GRPO 放弃了判别器模型，改为从组得分中估计基线，与近端策略优化 (PPO) 相比显著减少了训练资源。


- We demonstrate that GRPO significantly enhances the performance of our instruction-tuned model DeepSeekMath-Instruct, by solely using the instruction-tuning data. Furthermore, we observe enhancements in the out-of-domain performance during the reinforcement learning process.
- 我们证明了仅通过使用指令微调数据，GRPO 就能显著增强指令微调模型 DeepSeekMath-Instruct 的性能。此外，我们观察到在强化学习过程中，跨领域性能也有所提升。


- We provide a unified paradigm to understand different methods, such as RFT, DPO, PPO, and GRPO. We also conduct extensive experiments, e.g., online v.s. offline training, outcome v.s. process supervision, single-turn v.s. iterative reinforcement learning, and so on to deeply investigate the essential elements of this paradigm.
- 我们提供了一个统一范式来理解不同的方法，如 RFT、DPO、PPO 和 GRPO。我们还进行了广泛的实验（例如在线与离线训练、结果与过程监督、单轮与迭代强化学习等），以深入探究该范式的核心要素。


- Based on our unified paradigm, we explore the reasons behind the effectiveness of reinforcement learning, and summarize several potential directions to achieve more effective reinforcement learning of LLMs.
- 基于我们的统一范式，我们探讨了强化学习有效性的背后原因，并总结了实现更有效的 LLMs 强化学习的几个潜在方向。


### 1.2. Summary of Evaluations and Metrics
### 1.2. 评估与指标摘要


- English and Chinese Mathematical Reasoning: We conduct comprehensive assessments of our models on English and Chinese benchmarks, covering mathematical problems from grade-school level to college level. English benchmarks include GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), SAT (Azerbayev et al., 2023), OCW Courses (Lewkowycz et al., 2022a), MMLU-STEM (Hendrycks et al., 2020). Chinese benchmarks include MGSM-zh (Shi et al., 2023), CMATH (Wei et al., 2023), Gaokao-MathCloze (Zhong et al., 2023), and Gaokao-MathQA (Zhong et al., 2023). We evaluate models' ability to generate self-contained text solutions without tool use, and also the ability to solve problems using Python.
- 中英文数学推理：我们在中英文基准测试上对模型进行了全面评估，涵盖了从小学到大学水平的数学问题。英文基准包括 GSM8K (Cobbe et al., 2021)、MATH (Hendrycks et al., 2021)、SAT (Azerbayev et al., 2023)、OCW Courses (Lewkowycz et al., 2022a) 和 MMLU-STEM (Hendrycks et al., 2020)。中文基准包括 MGSM-zh (Shi et al., 2023)、CMATH (Wei et al., 2023)、Gaokao-MathCloze (Zhong et al., 2023) 和 Gaokao-MathQA (Zhong et al., 2023)。我们评估了模型在不使用工具的情况下生成独立文本解法的能力，以及使用 Python 解决问题的能力。


On English benchmarks, DeepSeekMath-Base is competitive with the closed-source Minerva 540B (Lewkowycz et al., 2022a), and surpasses all open-source base models (e.g., Mistral 7B (Jiang et al., 2023) and Llemma-34B (Azerbayev et al., 2023)), regardless of whether they've undergone math pre-training or not, often by a significant margin. Notably, DeepSeekMath-Base is superior on Chinese benchmarks, likely because we don't follow previous works (Azerbayev et al., 2023; Lewkowycz et al., 2022a) to collect English-only math pre-training data, and also include high-quality non-English ones. With mathematical instruction tuning and reinforcement learning, the resulting DeepSeekMath-Instruct and DeepSeekMath-RL demonstrate strong performance, obtaining an accuracy of over 50% on the competition-level MATH dataset for the first time within the open-source community.
在英文基准上，DeepSeekMath-Base 与闭源的 Minerva 540B (Lewkowycz et al., 2022a) 具有竞争力，并且无论是否经过数学预训练，它都超越了所有开源基座模型（如 Mistral 7B (Jiang et al., 2023) 和 Llemma-34B (Azerbayev et al., 2023)），且通常领先优势显著。值得注意的是，DeepSeekMath-Base 在中文基准上表现更优，这可能是因为我们没有遵循前人的工作 (Azerbayev et al., 2023; Lewkowycz et al., 2022a) 仅收集英文数学预训练数据，而是同时包含了高质量的非英文数据。通过数学指令微调和强化学习，最终得到的 DeepSeekMath-Instruct 和 DeepSeekMath-RL 表现强劲，在开源社区中首次在竞赛级 MATH 数据集上获得了超过 50% 的准确率。


- Formal Mathematics: We evaluate DeepSeekMath-Base using the informal-to-formal theorem proving task from (Jiang et al., 2022) on miniF2F (Zheng et al., 2021) with Isabelle (Wenzel et al., 2008) chosen to be the proof assistant. DeepSeekMath-Base demonstrates strong few-shot autoformalization performance.
- 形式化数学：我们使用来自 (Jiang et al., 2022) 的非形式化到形式化定理证明任务，在以 Isabelle (Wenzel et al., 2008) 为证明助手的 miniF2F (Zheng et al., 2021) 上评估了 DeepSeekMath-Base。DeepSeekMath-Base 展示了强大的少样本自动形式化能力。


- Natural Language Understanding, Reasoning, and Code: To build a comprehensive profile of models' general understanding, reasoning, and coding capabilities, we evaluate DeepSeekMath-Base on the Massive Multitask Language Understanding (MMLU) benchmark (Hendrycks et al. 2020) which encompasses 57 multiple-choice tasks covering diverse subjects, BIG-Bench Hard (BBH) (Suzgun et al., 2022) which consists of 23 challenging tasks that mostly require multi-step reasoning to solve, as well as HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021) which are widely used to evaluate code language models. Math pre-training benefits both language understanding and reasoning performance.
- 自然语言理解、推理与代码：为了全面了解模型的通用理解、推理和编码能力，我们在 MMLU 基准 (Hendrycks et al. 2020)（涵盖 57 个不同学科的选择题任务）、BBH 基准 (Suzgun et al., 2022)（包含 23 个大多需要多步推理的挑战性任务）以及广泛用于评估代码语言模型的 HumanEval (Chen et al., 2021) 和 MBPP (Austin et al., 2021) 上对 DeepSeekMath-Base 进行了评估。数学预训练同时有益于语言理解和推理性能。


## 2. Math Pre-Training
## 2. 数学预训练


### 2.1. Data Collection and Decontamination
### 2.1. 数据收集与去污染


In this section, we will outline the process of constructing the DeepSeekMath Corpus from Common Crawl. As depicted in Figure 2, we present an iterative pipeline that demonstrates how to systematically gather a large-scale mathematical corpus from Common Crawl, starting with a seed corpus (e.g., a small but high-quality collection of math-related dataset). It's worth noting that this approach is also applicable to other domains, such as coding.
在本节中，我们将概述从 Common Crawl 构建 DeepSeekMath 语料库的过程。如图 2 所示，我们展示了一个迭代流水线，演示了如何从种子语料库（例如小规模但高质量的数学相关数据集）开始，系统地从 Common Crawl 中收集大规模数学语料库。值得注意的是，这种方法也适用于编码等其他领域。


First, we choose OpenWebMath (Paster et al., 2023), a collection of high-quality mathematical web texts, as our initial seed corpus. Using this corpus, we train a fastText model (Joulin et al. 2016) to recall more OpenWebMath-like mathematical web pages. Specifically, we randomly select 500,000 data points from the seed corpus as positive training examples and another 500,000 web pages from Common Crawl as negative ones. We employ an open-source library for training, configuring the vector dimension to 256, learning rate to 0.1, the maximum length of word n-gram to 3 , the minimum number of word occurrences to 3 , and the number of training epochs to 3. To reduce the size of the original Common Crawl, we employ URL-based deduplication and near-deduplication techniques, resulting in 40B HTML web pages. We then recall mathematical web pages from deduplicated Common Crawl with the fastText model. To filter out low-quality mathematical content, we rank the collected pages according to their scores predicted by the fastText model, and only preserve the top-ranking ones. The volume of data preserved is assessed through pre-training experiments on the top 40B, 80B, 120B, and 160B tokens. In the first iteration, we choose to keep the top 40B tokens.
首先，我们选择高质量数学网络文本集 OpenWebMath (Paster et al., 2023) 作为初始种子语料库。利用该语料库，我们训练了一个 fastText 模型 (Joulin et al. 2016) 以召回更多类似 OpenWebMath 的数学网页。具体而言，我们从种子语料库中随机选取 500,000 条数据作为正样本，从 Common Crawl 中随机选取 500,000 个网页作为负样本。我们使用开源库进行训练，配置向量维度为 256，学习率为 0.1，词 n-gram 最大长度为 3，词频最小次数为 3，训练轮数为 3。为了减小原始 Common Crawl 的体积，我们采用基于 URL 的去重和近似去重技术，得到 400 亿个 HTML 网页。随后，我们利用 fastText 模型从去重后的 Common Crawl 中召回数学网页。为了过滤低质量数学内容，我们根据 fastText 模型预测的分数对收集到的页面进行排序，仅保留排名靠前的页面。通过对前 400 亿、800 亿、1200 亿和 1600 亿 token 进行预训练实验来评估保留的数据量。在第一次迭代中，我们选择保留前 400 亿 token。


---



https://fasttext.cc



---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_17_11_e8048c.jpg"/>



Figure 2 | An iterative pipeline that collects mathematical web pages from Common Crawl.
图 2 | 从 Common Crawl 收集数学网页的迭代流水线。


After the first iteration of data collection, numerous mathematical web pages remain uncollected, mainly because the fastText model is trained on a set of positive examples that lacks sufficient diversity. We therefore identify additional mathematical web sources to enrich the seed corpus, so that we can optimize the fastText model. Specifically, we first organize the entire Common Crawl into disjoint domains; a domain is defined as web pages sharing the same base URL. For each domain, we calculate the percentage of web pages that are collected in the first iteration. Domains where over 10% of the web pages have been collected are classified as math-related (e.g., mathoverflow.net). Subsequently, we manually annotate the URLs associated with mathematical content within these identified domains (e.g., mathoverflow.net/questions). Web pages linked to these URLs, yet uncollected, will be added to the seed corpus. This approach enables us to gather more positive examples, thereby training an improved fastText model capable of recalling more mathematical data in the subsequent iteration. After four iterations of data collection, we end up with 35.5M mathematical web pages, totaling 120B tokens. In the fourth iteration, we notice that nearly 98% of the data has already been collected in the third iteration, so we decide to cease data collection.
在第一轮数据收集后，仍有大量数学网页未被采集，这主要是因为 fastText 模型训练所用的正样本集缺乏足够的样性。因此，我们确定了额外的数学网络资源以丰富种子语料库，从而优化 fastText 模型。具体而言，我们首先将整个 Common Crawl 划分为互不重叠的域名；域名定义为共享相同基础 URL 的网页。对于每个域名，我们计算在第一轮迭代中收集到的网页百分比。如果收集比例超过 10%，该域名将被归类为数学相关域名（例如 mathoverflow.net）。随后，我们手动标注这些识别出的域名中与数学内容相关的 URL（例如 mathoverflow.net/questions）。与这些 URL 链接但尚未收集的网页将被添加到种子语料库中。这种方法使我们能够收集更多正样本，从而训练出改进的 fastText 模型，以便在后续迭代中召回更多数学数据。经过四轮数据收集，我们最终获得了 3550 万个数学网页，共计 1200 亿 token。在第四轮迭代中，我们注意到近 98% 的数据已在第三轮中收集完毕，因此决定停止数据收集。


To avoid benchmark contamination, we follow Guo et al. (2024) to filter out web pages containing questions or answers from English mathematical benchmarks such as GSM8K (Cobbe et al. 2021) and MATH (Hendrycks et al. 2021) and Chinese benchmarks such as CMATH (Wei et al., 2023) and AGIEval (Zhong et al., 2023). The filtering criteria are as follows: any text segment containing a 10-gram string that matches exactly with any sub-string from the evaluation benchmarks is removed from our math training corpus. For benchmark texts that are shorter than 10 grams but have at least 3 grams, we employ exact matching to filter out contaminated web pages.
为避免基准测试污染，我们遵循 Guo et al. (2024) 的方法，过滤掉包含英语数学基准测试（如 GSM8K (Cobbe et al. 2021) 和 MATH (Hendrycks et al. 2021)）以及中文基准测试（如 CMATH (Wei et al., 2023) 和 AGIEval (Zhong et al., 2023)）中问题或答案的网页。过滤标准如下：任何包含与评估基准测试中任一子串完全匹配的 10-gram 字符串的文本段均从数学训练语料库中删除。对于短于 10-gram 但至少有 3-gram 的基准测试文本，我们采用精确匹配来过滤受污染的网页。


### 2.2. Validating the Quality of the DeepSeekMath Corpus
### 2.2. 验证 DeepSeekMath 语料库的质量


We run pre-training experiments to investigate how the DeepSeekMath Corpus is compared with the recently released math-training corpora:
我们进行了预训练实验，以研究 DeepSeekMath 语料库与近期发布的数学训练语料库的对比情况：


- MathPile (Wang et al., 2023c): a multi-source corpus (8.9B tokens) aggregated from textbooks, Wikipedia, ProofWiki, CommonCrawl, StackExchange, and arXiv, with the majority (over 85%) sourced from arXiv;
- MathPile (Wang et al., 2023c)：一个包含教材、维基百科、ProofWiki、CommonCrawl、StackExchange 和 arXiv 的多源语料库（8.9B token），其中大部分（超过 85%）源自 arXiv；


- OpenWebMath (Paster et al. 2023): CommonCrawl data filtered for mathematical content, totaling 13.6B tokens;
- OpenWebMath (Paster et al. 2023)：经过数学内容过滤的 CommonCrawl 数据，共计 13.6B token；


- Proof-Pile-2 (Azerbayev et al. 2023): a mathematical corpus consisting of OpenWeb-Math, AlgebraicStack (10.3B tokens of mathematical code), and arXiv papers (28.0B tokens). When experimenting on Proof-Pile-2, we follow Azerbayev et al. (2023) to use an arXiv:Web:Code ratio of 2:4:1.
- Proof-Pile-2 (Azerbayev et al. 2023)：一个由 OpenWebMath、AlgebraicStack（10.3B token 的数学代码）和 arXiv 论文（28.0B token）组成的数学语料库。在 Proof-Pile-2 上进行实验时，我们遵循 Azerbayev et al. (2023) 的做法，使用 2:4:1 的 arXiv:Web:Code 比例。


#### 2.2.1. Training Setting
#### 2.2.1. 训练设置


We apply math training to a general pre-trained language model with 1.3B parameters, which shares the same framework as the DeepSeek LLMs (DeepSeek-AI, 2024), denoted as DeepSeek-LLM 1.3B. We separately train a model on each mathematical corpus for 150B tokens. All experiments are conducted using the efficient and light-weight HAI-LLM (High-flyer, 2023) training framework. Following the training practice of DeepSeek LLMs, we use the AdamW optimizer (Loshchilov and Hutter, 2017) with ${\beta }_{1} = {0.9},{\beta }_{2} = {0.95}$ ,and weight_decay $= {0.1}$ ,along with a multi-step learning rate schedule where the learning rate reaches the peak after 2,000 warmup steps, decreases to its 31.6% after 80% of the training process, and further decreases to 10.0% of the peak after 90% of the training process. We set the maximum value of learning rate to 5.3e-4, and use a batch size of 4M tokens with a 4K context length.
我们将数学训练应用于一个具有1.3B参数的通用预训练语言模型，该模型与DeepSeek LLMs (DeepSeek-AI, 2024) 架构相同，记作DeepSeek-LLM 1.3B。我们在每个数学语料库上分别训练150B tokens。所有实验均使用高效轻量化的HAI-LLM (High-flyer, 2023) 训练框架。遵循DeepSeek LLMs的训练实践，我们使用AdamW优化器 (Loshchilov and Hutter, 2017)，其中 ${\beta }_{1} = {0.9},{\beta }_{2} = {0.95}$，权重衰减为 $= {0.1}$，并采用多步学习率调度：学习率在2,000步预热后达到峰值，在训练进度的80%时降至峰值的31.6%，在90%时进一步降至10.0%。我们将学习率最大值设定为5.3e-4，并使用4M tokens的批大小和4K上下文长度。


<table><tr><td rowspan="2">Math Corpus</td><td rowspan="2">Size</td><td colspan="5">English Benchmarks</td><td colspan="3">Chinese Benchmarks</td></tr><tr><td>GSM8K</td><td>MATH</td><td>OCW</td><td>SAT</td><td>MMLU STEM</td><td>CMATH</td><td>Gaokao MathCloze</td><td>Gaokao MathQA</td></tr><tr><td>No Math Training</td><td>N/A</td><td>2.9%</td><td>3.0%</td><td>2.9%</td><td>15.6%</td><td>19.5%</td><td>12.3%</td><td>0.8%</td><td>17.9%</td></tr><tr><td>MathPile</td><td>8.9B</td><td>2.7%</td><td>3.3%</td><td>2.2%</td><td>12.5%</td><td>15.7%</td><td>1.2%</td><td>0.0%</td><td>2.8%</td></tr><tr><td>OpenWebMath</td><td>13.6B</td><td>11.5%</td><td>8.9%</td><td>3.7%</td><td>31.3%</td><td>29.6%</td><td>16.8%</td><td>0.0%</td><td>14.2%</td></tr><tr><td>Proof-Pile-2</td><td>51.9B</td><td>14.3%</td><td>11.2%</td><td>3.7%</td><td>43.8%</td><td>29.2%</td><td>19.9%</td><td>5.1%</td><td>11.7%</td></tr><tr><td>DeepSeekMath Corpus</td><td>120.2B</td><td>23.8%</td><td>13.6%</td><td>4.8%</td><td>56.3%</td><td>33.1%</td><td>41.5%</td><td>5.9%</td><td>23.6%</td></tr></table>
<table><tbody><tr><td rowspan="2">数学语料库</td><td rowspan="2">规模</td><td colspan="5">英文基准测试</td><td colspan="3">中文基准测试</td></tr><tr><td>GSM8K</td><td>MATH</td><td>OCW</td><td>SAT</td><td>MMLU STEM</td><td>CMATH</td><td>高考数学填空题</td><td>高考数学选择题</td></tr><tr><td>无数学训练</td><td>不适用</td><td>2.9%</td><td>3.0%</td><td>2.9%</td><td>15.6%</td><td>19.5%</td><td>12.3%</td><td>0.8%</td><td>17.9%</td></tr><tr><td>MathPile</td><td>8.9B</td><td>2.7%</td><td>3.3%</td><td>2.2%</td><td>12.5%</td><td>15.7%</td><td>1.2%</td><td>0.0%</td><td>2.8%</td></tr><tr><td>OpenWebMath</td><td>13.6B</td><td>11.5%</td><td>8.9%</td><td>3.7%</td><td>31.3%</td><td>29.6%</td><td>16.8%</td><td>0.0%</td><td>14.2%</td></tr><tr><td>Proof-Pile-2</td><td>51.9B</td><td>14.3%</td><td>11.2%</td><td>3.7%</td><td>43.8%</td><td>29.2%</td><td>19.9%</td><td>5.1%</td><td>11.7%</td></tr><tr><td>DeepSeekMath 语料库</td><td>120.2B</td><td>23.8%</td><td>13.6%</td><td>4.8%</td><td>56.3%</td><td>33.1%</td><td>41.5%</td><td>5.9%</td><td>23.6%</td></tr></tbody></table>


Table 1 | Performance of DeepSeek-LLM 1.3B trained on different mathematical corpora, evaluated using few-shot chain-of-thought prompting. Corpus sizes are calculated using our tokenizer with a vocabulary size of ${100}\mathrm{\;K}$ .
表 1 | 在不同数学语料库上训练的 DeepSeek-LLM 1.3B 的性能，使用 few-shot 思维链提示进行评估。语料库大小使用词表大小为 ${100}\mathrm{\;K}$ 的分词器计算。


#### 2.2.2. Evaluation Results
#### 2.2.2. 评估结果


The DeepSeekMath Corpus is of high quality, covers multilingual mathematical content, and is the largest in size.
DeepSeekMath 语料库质量高，涵盖多语言数学内容，且规模最大。


- High-quality: We evaluate downstream performance on 8 mathematical benchmarks using few-shot chain-of-thought prompting Wei et al. (2022). As shown in Table 1, there is a clear performance lead of the model trained on the DeepSeekMath Corpus. Figure 3 shows that the model trained on the DeepSeekMath Corpus demonstrates better performance than Proof-Pile-2 at 50B tokens (1 full epoch of Proof-Pile-2), indicating the average quality of DeepSeekMath Corpus is higher.
- 高质量：我们使用 few-shot 思维链提示（Wei 等，2022）评估了 8 个数学基准测试的下游性能。如表 1 所示，在 DeepSeekMath 语料库上训练的模型具有明显的性能领先优势。图 3 显示，在 DeepSeekMath 语料库上训练的模型在 50B token（Proof-Pile-2 的 1 个完整 epoch）处表现出比 Proof-Pile-2 更好的性能，表明 DeepSeekMath 语料库的平均质量更高。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_17_11_77421d.jpg"/>



Figure 3 | Benchmark curves of DeepSeek-LLM 1.3B trained on different mathematical corpora.
图 3 | 在不同数学语料库上训练的 DeepSeek-LLM 1.3B 的基准测试曲线。


- Multilingual: The DeepSeekMath Corpus encompasses data in multiple languages, predominantly featuring English and Chinese as the two most represented languages. As shown in Table 1, training on the DeepSeekMath Corpus enhances mathematical reasoning performance in both English and Chinese. In contrast, existing mathematical corpora, which are primarily English-centric, show limited improvement and may even hinder performance in Chinese mathematical reasoning.
- 多语言：DeepSeekMath 语料库包含多种语言的数据，其中英文和中文是代表性最强的两种语言。如表 1 所示，在 DeepSeekMath 语料库上进行训练增强了中英文的数学推理性能。相比之下，现有的以英文为核心的数学语料库提升有限，甚至可能阻碍中文数学推理性能。


- Large-scale: The DeepSeekMath Corpus is several times larger than existing mathematical corpora. As depicted in Figure 3. DeepSeek-LLM 1.3B, when trained on the DeepSeek-Math Corpus, shows a steeper learning curve along with more lasting improvements. In contrast, the baseline corpora are much smaller, and have already been repeated multiple rounds during training, with the resulting model performance quickly reaching a plateau.
- 大规模：DeepSeekMath 语料库规模比现有数学语料库大数倍。如图 3 所示，DeepSeek-LLM 1.3B 在 DeepSeekMath 语料库上训练时，表现出更陡峭的学习曲线和更持久的提升。相比之下，基准语料库规模要小得多，在训练期间已经重复了多轮，导致模型性能迅速达到瓶颈。


### 2.3. Training and Evaluating DeepSeekMath-Base 7B
### 2.3. 训练与评估 DeepSeekMath-Base 7B


In this section, we introduce DeepSeekMath-Base 7B, a base model with strong reasoning abilities, especially in mathematics. Our model is initialized with DeepSeek-Coder-Base-v1.5 7B
在本节中，我们介绍 DeepSeekMath-Base 7B，这是一个具有强大推理能力（尤其是数学能力）的基础模型。我们的模型以 DeepSeek-Coder-Base-v1.5 7B 为基础进行初始化


(Guo et al. 2024) and trained for 500B tokens. The distribution of the data is as follows: 56% is from the DeepSeekMath Corpus, 4% from AlgebraicStack, 10% from arXiv, 20% is Github code, and the remaining 10% is natural language data from Common Crawl in both English and Chinese. We mainly adopt the training setting specified in Section 2.2.1, except that we set the maximum value of the learning rate to 4.2e-4 and use a batch size of ${10}\mathrm{M}$ tokens.
（Guo 等，2024）并训练了 500B token。数据分布如下：56% 来自 DeepSeekMath 语料库，4% 来自 AlgebraicStack，10% 来自 arXiv，20% 是 GitHub 代码，剩余 10% 是来自 Common Crawl 的中英文自然语言数据。我们主要采用 2.2.1 节中指定的训练设置，不同之处在于我们将学习率最大值设为 4.2e-4，并使用 ${10}\mathrm{M}$ token 的 batch size。


We conduct a comprehensive assessment of the mathematical capabilities of DeepSeekMath-Base 7B, focusing on its ability to produce self-contained mathematical solutions without relying on external tools, solve mathematical problems using tools, and conduct formal theorem proving. Beyond mathematics, we also provide a more general profile of the base model, including its performance of natural language understanding, reasoning, and programming skills.
我们对 DeepSeekMath-Base 7B 的数学能力进行了全面评估，重点关注其在不依赖外部工具的情况下生成独立数学解答、利用工具解决数学问题以及进行形式化定理证明的能力。除数学之外，我们还提供了该基础模型更通用的概况，包括其在自然语言理解、推理和编程技能方面的表现。


Mathematical Problem Solving with Step-by-Step Reasoning We evaluate DeepSeekMath-Base's performance of solving mathematical problems using few-shot chain-of-thought prompting (Wei et al. 2022), across eight benchmarks in English and Chinese. These benchmarks encompass quantitative reasoning (e.g., GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), and CMATH (Wei et al., 2023)) and multiple-choice problems (e.g., MMLU-STEM (Hendrycks et al., 2020) and Gaokao-MathQA (Zhong et al., 2023)), covering diverse fields of mathematics from elementary to college-level complexity.
使用分步推理解决数学问题 我们使用 few-shot 思维链提示（Wei 等，2022）在八个中英文基准测试中评估了 DeepSeekMath-Base 解决数学问题的性能。这些基准测试涵盖了定量推理（例如 GSM8K (Cobbe et al., 2021)、MATH (Hendrycks et al., 2021) 和 CMATH (Wei et al., 2023)）和多选题（例如 MMLU-STEM (Hendrycks et al., 2020) 和 Gaokao-MathQA (Zhong et al., 2023)），涵盖了从小学到大学难度的各种数学领域。


As shown in Table 2, DeepSeekMath-Base 7B leads in performance across all eight benchmarks among the open-source base models (including the widely-used general model Mistral 7B (Jiang et al. 2023) and the recently released Llemma 34B (Azerbayev et al. 2023) which underwent math training on Proof-Pile-2 (Azerbayev et al., 2023)). Notably, on the competition-level MATH dataset, DeepSeekMath-Base surpasses existing open-source base models by over 10% absolute, and outperforms Minerva 540B (Lewkowycz et al., 2022a), a closed-source base model 77 times larger which builds on PaLM (Lewkowycz et al., 2022b) and is further trained on mathematical texts.
如表 2 所示，DeepSeekMath-Base 7B 在所有八个基准测试中的表现均领先于开源基础模型（包括广泛使用的通用模型 Mistral 7B (Jiang et al. 2023) 和最近发布的在 Proof-Pile-2 (Azerbayev et al., 2023) 上进行过数学训练的 Llemma 34B (Azerbayev et al. 2023)）。值得注意的是，在竞赛级 MATH 数据集上，DeepSeekMath-Base 超过现有开源基础模型 10% 以上，并优于 Minerva 540B (Lewkowycz et al., 2022a)，后者是一个规模大 77 倍、基于 PaLM (Lewkowycz et al., 2022b) 并进一步在数学文本上训练的闭源基础模型。


<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="5">English Benchmarks</td><td colspan="3">Chinese Benchmarks</td></tr><tr><td>GSM8K</td><td>MATH</td><td>OCW</td><td>SAT</td><td>MMLU STEM</td><td>CMATH</td><td>Gaokao MathCloze</td><td>Gaokao MathQA</td></tr><tr><td colspan="10">Closed-Source Base Model</td></tr><tr><td>Minerva</td><td>7B</td><td>16.2%</td><td>14.1%</td><td>7.7%</td><td>-</td><td>35.6%</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Minerva</td><td>62B</td><td>52.4%</td><td>27.6%</td><td>12.0%</td><td>-</td><td>53.9%</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Minerva</td><td>540B</td><td>58.8%</td><td>33.6%</td><td>17.6%</td><td>-</td><td>63.9%</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="10">Open-Source Base Model</td></tr><tr><td>Mistral</td><td>7B</td><td>40.3%</td><td>14.3%</td><td>9.2%</td><td>71.9%</td><td>51.1%</td><td>44.9%</td><td>5.1%</td><td>23.4%</td></tr><tr><td>Llemma</td><td>7B</td><td>37.4%</td><td>18.1%</td><td>6.3%</td><td>59.4%</td><td>43.1%</td><td>43.4%</td><td>11.9%</td><td>23.6%</td></tr><tr><td>Llemma</td><td>34B</td><td>54.0%</td><td>25.3%</td><td>10.3%</td><td>71.9%</td><td>52.9%</td><td>56.1%</td><td>11.9%</td><td>26.2%</td></tr><tr><td>DeepSeekMath-Base 7B</td><td></td><td>64.2%</td><td>36.2%</td><td>15.4%</td><td>84.4%</td><td>56.5%</td><td>71.7%</td><td>20.3%</td><td>35.3%</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">尺寸</td><td colspan="5">英文基准</td><td colspan="3">中文基准</td></tr><tr><td>GSM8K</td><td>MATH</td><td>OCW</td><td>SAT</td><td>MMLU STEM</td><td>CMATH</td><td>高考数学填空</td><td>高考数学选择</td></tr><tr><td colspan="10">闭源基座模型</td></tr><tr><td>Minerva</td><td>7B</td><td>16.2%</td><td>14.1%</td><td>7.7%</td><td>-</td><td>35.6%</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Minerva</td><td>62B</td><td>52.4%</td><td>27.6%</td><td>12.0%</td><td>-</td><td>53.9%</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Minerva</td><td>540B</td><td>58.8%</td><td>33.6%</td><td>17.6%</td><td>-</td><td>63.9%</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="10">开源基座模型</td></tr><tr><td>Mistral</td><td>7B</td><td>40.3%</td><td>14.3%</td><td>9.2%</td><td>71.9%</td><td>51.1%</td><td>44.9%</td><td>5.1%</td><td>23.4%</td></tr><tr><td>Llemma</td><td>7B</td><td>37.4%</td><td>18.1%</td><td>6.3%</td><td>59.4%</td><td>43.1%</td><td>43.4%</td><td>11.9%</td><td>23.6%</td></tr><tr><td>Llemma</td><td>34B</td><td>54.0%</td><td>25.3%</td><td>10.3%</td><td>71.9%</td><td>52.9%</td><td>56.1%</td><td>11.9%</td><td>26.2%</td></tr><tr><td>DeepSeekMath-Base 7B</td><td></td><td>64.2%</td><td>36.2%</td><td>15.4%</td><td>84.4%</td><td>56.5%</td><td>71.7%</td><td>20.3%</td><td>35.3%</td></tr></tbody></table>


Table 2 | Comparisons between DeepSeekMath-Base 7B and strong base models on English and Chinese mathematical benchmarks. Models are evaluated with chain-of-thought prompting. Minerva results are quoted from Lewkowycz et al. (2022a).
表 2 | DeepSeekMath-Base 7B 与强基座模型在英文和中文数学基准测试上的对比。模型采用思维链（CoT）提示进行评估。Minerva 的结果引用自 Lewkowycz et al. (2022a)。


Mathematical Problem Solving with Tool Use We evaluate program-aided mathematical reasoning on GSM8K and MATH using few-shot program-of-thought prompting (Chen et al. 2022; Gao et al., 2023). Models are prompted to solve each problem by writing a Python program where libraries such as math and sympy can be utilized for intricate computations. The execution result of the program is evaluated as the answer. As shown in Table 3, DeepSeekMath-Base 7B outperforms the prior state-of-the-art Llemma 34B.
结合工具使用的数学问题求解 我们使用少样本程序思维（PoT）提示（Chen et al. 2022; Gao et al., 2023）在 GSM8K 和 MATH 上评估程序辅助数学推理。模型通过编写 Python 程序来解决问题，并可利用 math 和 sympy 等库进行复杂计算。程序的执行结果即为答案。如表 3 所示，DeepSeekMath-Base 7B 的表现优于之前的先进模型 Llemma 34B。


<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="2">Problem Solving w/ Tools</td><td colspan="2">Informal-to-Formal Proving</td></tr><tr><td>GSM8K+Python</td><td>MATH+Python</td><td>miniF2F-valid</td><td>miniF2F-test</td></tr><tr><td>Mistral</td><td>7B</td><td>48.5%</td><td>18.2%</td><td>18.9%</td><td>18.0%</td></tr><tr><td>CodeLlama</td><td>7B</td><td>27.1%</td><td>17.2%</td><td>16.3%</td><td>17.6%</td></tr><tr><td>CodeLlama</td><td>34B</td><td>52.7%</td><td>23.5%</td><td>18.5%</td><td>18.0%</td></tr><tr><td>Llemma</td><td>7B</td><td>41.0%</td><td>18.6%</td><td>20.6%</td><td>22.1%</td></tr><tr><td>Llemma</td><td>34B</td><td>64.6%</td><td>26.3%</td><td>21.0%</td><td>21.3%</td></tr><tr><td>DeepSeekMath-Base 7B</td><td></td><td>66.9%</td><td>31.4%</td><td>25.8%</td><td>$\mathbf{{24.6}\% }$</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">尺寸</td><td colspan="2">利用工具解决问题</td><td colspan="2">非形式化到形式化证明</td></tr><tr><td>GSM8K+Python</td><td>MATH+Python</td><td>miniF2F-valid</td><td>miniF2F-test</td></tr><tr><td>Mistral</td><td>7B</td><td>48.5%</td><td>18.2%</td><td>18.9%</td><td>18.0%</td></tr><tr><td>CodeLlama</td><td>7B</td><td>27.1%</td><td>17.2%</td><td>16.3%</td><td>17.6%</td></tr><tr><td>CodeLlama</td><td>34B</td><td>52.7%</td><td>23.5%</td><td>18.5%</td><td>18.0%</td></tr><tr><td>Llemma</td><td>7B</td><td>41.0%</td><td>18.6%</td><td>20.6%</td><td>22.1%</td></tr><tr><td>Llemma</td><td>34B</td><td>64.6%</td><td>26.3%</td><td>21.0%</td><td>21.3%</td></tr><tr><td>DeepSeekMath-Base 7B</td><td></td><td>66.9%</td><td>31.4%</td><td>25.8%</td><td>$\mathbf{{24.6}\% }$</td></tr></tbody></table>


Table 3 | Few-shot evaluation of base models' ability to solve mathematical problems using tools and the ability to conduct informal-to-formal theorem proving in Isabelle.
表 3 | 基础模型利用工具解决数学问题以及在 Isabelle 中进行非正式到正式定理证明能力的少样本评估。


Formal Mathematics Formal proof automation is beneficial to ensure the accuracy and reliability of mathematical proofs and enhance efficiency, with increasing attention in recent years. We evaluate DeepSeekMath-Base 7B on the task of informal-to-formal proving from (Jiang et al. 2022) which is to generate a formal proof based on an informal statement, a formal counterpart of the statement, and an informal proof. We evaluate on miniF2F (Zheng et al., 2021), a benchmark for formal Olympiad-level mathematics, and generate a formal proof in Isabelle for each problem with few-shot prompting. Following Jiang et al. (2022), we leverage models to generate proof sketches, and execute the off-the-shelf automated prover Sledgehammer (Paulson, 2010) to fill in the missing details. As shown in Table 3, DeepSeekMath-Base 7B demonstrates strong performance in proof autoformalization.
形式化数学 形式化证明自动化有利于确保数学证明的准确性和可靠性并提高效率，近年来受到越来越多的关注。我们评估了 DeepSeekMath-Base 7B 在 (Jiang et al. 2022) 提出的非正式到正式证明任务上的表现，该任务旨在根据非正式陈述、陈述的形式化对应版本以及非正式证明生成形式化证明。我们在奥数级形式化数学基准测试 miniF2F (Zheng et al., 2021) 上进行评估，并通过少样本提示为每个问题生成 Isabelle 形式化证明。遵循 Jiang et al. (2022) 的方法，我们利用模型生成证明草图，并运行现成的自动化证明器 Sledgehammer (Paulson, 2010) 来填充缺失的细节。如表 3 所示，DeepSeekMath-Base 7B 在证明自动形式化方面表现出强劲的性能。


<table><tr><td>Model</td><td>Size</td><td>MMLU</td><td>BBH</td><td>HumanEval (Pass@1)</td><td>MBPP (Pass@1)</td></tr><tr><td>Mistral</td><td>7B</td><td>62.4%</td><td>55.7%</td><td>28.0%</td><td>41.4%</td></tr><tr><td>DeepSeek-Coder-Base-v1.5†</td><td>7B</td><td>42.9%</td><td>42.9%</td><td>40.2%</td><td>52.6%</td></tr><tr><td>DeepSeek-Coder-Base-v1.5</td><td>7B</td><td>49.1%</td><td>55.2%</td><td>43.2%</td><td>60.4%</td></tr><tr><td>DeepSeekMath-Base</td><td>7B</td><td>54.9%</td><td>59.5%</td><td>40.9%</td><td>52.6%</td></tr></table>
<table><tbody><tr><td>模型</td><td>尺寸</td><td>MMLU</td><td>BBH</td><td>HumanEval (Pass@1)</td><td>MBPP (Pass@1)</td></tr><tr><td>Mistral</td><td>7B</td><td>62.4%</td><td>55.7%</td><td>28.0%</td><td>41.4%</td></tr><tr><td>DeepSeek-Coder-Base-v1.5†</td><td>7B</td><td>42.9%</td><td>42.9%</td><td>40.2%</td><td>52.6%</td></tr><tr><td>DeepSeek-Coder-Base-v1.5</td><td>7B</td><td>49.1%</td><td>55.2%</td><td>43.2%</td><td>60.4%</td></tr><tr><td>DeepSeekMath-Base</td><td>7B</td><td>54.9%</td><td>59.5%</td><td>40.9%</td><td>52.6%</td></tr></tbody></table>


Table 4 | Evaluation on natural language understanding, reasoning, and code benchmarks. DeepSeek-Coder-Base-v1.5 ${}^{ \dagger  }$ is the checkpoint right before learning rate decay,which is used to train DeepSeekMath-Base. On MMLU and BBH, we use few-shot chain-of-thought prompting. On HumanEval and MBPP, we evaluate model performance under the zero-shot setting and a few-shot setting, respectively.
表 4 | 自然语言理解、推理和代码基准测试评估。DeepSeek-Coder-Base-v1.5 ${}^{ \dagger  }$ 是学习率衰减前的检查点，用于训练 DeepSeekMath-Base。在 MMLU 和 BBH 上，我们使用 few-shot 思维链提示。在 HumanEval 和 MBPP 上，我们分别在 zero-shot 和 few-shot 设置下评估模型性能。


Natural Language Understanding, Reasoning, and Code We evaluate model performance of natural language understanding on MMLU (Hendrycks et al., 2020), reasoning on BBH (Suzgun et al. 2022), and coding capabilities on HumanEval (Chen et al., 2021) and MBPP (Austin et al. 2021). As shown in Table 4, DeepSeekMath-Base 7B exhibits significant enhancements in performance on MMLU and BBH over its precursor, DeepSeek-Coder-Base-v1.5 (Guo et al., 2024), illustrating the positive impact of math training on language understanding and reasoning. Additionally, by including code tokens for continual training, DeepSeekMath-Base 7B effectively maintains the performance of DeepSeek-Coder-Base-v1.5 on the two coding benchmarks. Overall, DeepSeekMath-Base 7B significantly outperforms the general model Mistral 7B (Jiang et al., 2023) on the three reasoning and coding benchmarks.
自然语言理解、推理与代码 我们在 MMLU（Hendrycks et al., 2020）上评估自然语言理解性能，在 BBH（Suzgun et al. 2022）上评估推理性能，在 HumanEval（Chen et al., 2021）和 MBPP（Austin et al. 2021）上评估代码能力。如表 4 所示，DeepSeekMath-Base 7B 在 MMLU 和 BBH 上的表现较其前身 DeepSeek-Coder-Base-v1.5（Guo et al., 2024）有显著提升，说明了数学训练对语言理解和推理的积极影响。此外，通过引入代码 token 进行持续训练，DeepSeekMath-Base 7B 在两个代码基准测试上有效保持了 DeepSeek-Coder-Base-v1.5 的性能。总体而言，DeepSeekMath-Base 7B 在三个推理和代码基准测试上显著优于通用模型 Mistral 7B（Jiang et al., 2023）。


## 3. Supervised Fine-Tuning
## 3. 有监督微调


#### 3.1.SFT Data Curation
#### 3.1. SFT 数据策划分选


We construct a mathematical instruction-tuning dataset covering English and Chinese problems from different mathematical fields and of varying complexity levels: problems are paired with solutions in chain-of-thought (CoT) (Wei et al., 2022), program-of-thought (PoT) (Chen et al., 2022; Gao et al. 2023), and tool-integrated reasoning format (Gou et al., 2023). The total number of training examples is ${776}\mathrm{\;K}$ .
我们构建了一个数学指令微调数据集，涵盖了来自不同数学领域、具有不同复杂程度的中英文问题：问题与思维链（CoT）（Wei et al., 2022）、程序思维（PoT）（Chen et al., 2022; Gao et al. 2023）以及工具集成推理格式（Gou et al., 2023）的解决方案配对。训练样本总数为 ${776}\mathrm{\;K}$。


- English mathematical datasets: We annotate GSM8K and MATH problems with tool-integrated solutions, and adopt a subset of MathInstruct (Yue et al., 2023) along with the training set of Lila-OOD (Mishra et al., 2022) where problems are solved with CoT or PoT. Our English collection covers diverse fields of mathematics, e.g., algebra, probability, number theory, calculus, and geometry.
- 英文数学数据集：我们为 GSM8K 和 MATH 问题标注了工具集成解决方案，并采用了 MathInstruct（Yue et al., 2023）的一个子集以及 Lila-OOD（Mishra et al., 2022）的训练集，其中的问题通过 CoT 或 PoT 解决。我们的英文集合涵盖了代数、概率、数论、微积分和几何等多个数学领域。


- Chinese mathematical datasets: We collect Chinese K-12 mathematical problems spanning 76 sub-topics such as linear equations, with solutions annotated in both CoT and tool-integrated reasoning format.
- 中文数学数据集：我们收集了涵盖线性方程等 76 个子话题的中文 K-12 数学问题，解决方案均以 CoT 和工具集成推理格式进行了标注。


### 3.2. Training and Evaluating DeepSeekMath-Instruct 7B
### 3.2. 训练并评估 DeepSeekMath-Instruct 7B


In this section, we introduce DeepSeekMath-Instruct 7B which undergoes mathematical instruction tuning based on DeepSeekMath-Base. Training examples are randomly concatenated until reaching a maximum context length of $4\mathrm{\;K}$ tokens. We train the model for 500 steps with a batch size of 256 and a constant learning rate of 5e-5.
在本节中，我们介绍基于 DeepSeekMath-Base 进行数学指令微调的 DeepSeekMath-Instruct 7B。训练样本随机拼接，直到达到 $4\mathrm{\;K}$ token 的最大上下文长度。我们使用 256 的批次大小和 5e-5 的恒定学习率对模型进行了 500 步训练。


We evaluate models' mathematical performance both without and with tool use, on 4 quantitative reasoning benchmarks in English and Chinese. We benchmark our model against the leading models of the time:
我们在 4 个中英文定量推理基准测试上，分别评估了模型在不使用和使用工具情况下的数学性能。我们将模型与当时的领先模型进行了基准测试：


- Closed-source models include: (1) the GPT family among which GPT-4 (OpenAI, 2023) and GPT-4 Code Interpreter ${}^{2}$ are the most capable ones,(2) Gemini Ultra and Pro (Anil et al., 2023), (3) Inflection-2 (Inflection AI, 2023), (4) Grok-1 ${}^{3}$ ) as well as models recently released by Chinese companies including (5) Baichuan-3 (6) the latest GLM-4 5 from the GLM family (Du et al., 2022). These models are for general purposes, most of which have undergone a series of alignment procedures.
- 闭源模型包括：(1) GPT 系列，其中 GPT-4（OpenAI, 2023）和 GPT-4 Code Interpreter ${}^{2}$ 是能力最强的，(2) Gemini Ultra 和 Pro（Anil et al., 2023），(3) Inflection-2（Inflection AI, 2023），(4) Grok-1 ${}^{3}$ 以及中国公司最近发布的模型，包括 (5) 百川-3 (6) GLM 系列中最新的 GLM-4 5（Du et al., 2022）。这些模型均为通用目的，且大多经历了一系列对齐流程。


- Open-source models include: general models like (1) DeepSeek-LLM-Chat 67B (DeepSeek-AI, 2024), (2) Qwen 72B (Bai et al., 2023), (3) SeaLLM-v2 7B (Nguyen et al., 2023), and (4)
- 开源模型包括：通用模型如 (1) DeepSeek-LLM-Chat 67B（DeepSeek-AI, 2024），(2) 通义千问 Qwen 72B（Bai et al., 2023），(3) SeaLLM-v2 7B（Nguyen et al., 2023），以及 (4)


---



${}^{2}$ https://openai.com/blog/chatgpt-plugins#code-interpreter
${}^{2}$ https://openai.com/blog/chatgpt-plugins#code-interpreter


${}^{3}$ https://x.ai/model-card
${}^{3}$ https://x.ai/model-card


${}^{4}$ https://www.baichuan-ai.com
${}^{4}$ https://www.baichuan-ai.com


${}^{5}$ https://open.bigmodel.cn/dev/api#glm-4
${}^{5}$ https://open.bigmodel.cn/dev/api#glm-4


---



ChatGLM3 6B (ChatGLM3 Team, 2023), as well as models with enhancements in mathematics including (5) InternLM2-Math 20B ${}^{6}$ which builds on InternLM2 and underwent math training followed by instruction tuning, (6) Math-Shepherd-Mistral 7B which applys PPO training (Schulman et al. 2017) to Mistral 7B (Jiang et al. 2023) with a process-supervised reward model, (7) the WizardMath series (Luo et al. 2023) which improves mathematical reasoning in Mistral 7B and Llama-2 70B (Touvron et al., 2023) using evolve-instruct (i.e., a version of instruction tuning that uses AI-evolved instructions) and PPO training with training problems primarily sourced from GSM8K and MATH, (8) MetaMath 70B (Yu et al. 2023) which is Llama-2 70B fine-tuned on an augmented version of GSM8K and MATH, (9) ToRA 34B Gou et al. (2023) which is CodeLlama 34B fine-tuned to do tool-integrated mathematical reasoning, (10) MAmmoTH 70B (Yue et al. 2023) which is Llama-2 70B instruction-tuned on MathInstruct.
ChatGLM3 6B (ChatGLM3 Team, 2023)，以及在数学方面有所增强的模型，包括：(5) InternLM2-Math 20B ${}^{6}$，其基于 InternLM2 构建，经过数学训练及指令微调；(6) Math-Shepherd-Mistral 7B，对 Mistral 7B (Jiang et al. 2023) 应用了基于过程监督奖励模型的 PPO 训练 (Schulman et al. 2017)；(7) WizardMath 系列 (Luo et al. 2023)，利用 evolve-instruct（即使用 AI 进化指令的指令微调版本）和基于 GSM8K 与 MATH 题库的 PPO 训练，提升了 Mistral 7B 和 Llama-2 70B (Touvron et al., 2023) 的数学推理能力；(8) MetaMath 70B (Yu et al. 2023)，是在 GSM8K 和 MATH 增强版上微调的 Llama-2 70B；(9) ToRA 34B Gou et al. (2023)，是为工具集成数学推理而微调的 CodeLlama 34B；(10) MAmmoTH 70B (Yue et al. 2023)，是在 MathInstruct 上经过指令微调的 Llama-2 70B。


As shown in Table 5, under the evaluation setting where tool use is disallowed, DeepSeekMath-Instruct 7B demonstrates strong performance of step-by-step reasoning. Notably, on the competition-level MATH dataset, our model surpasses all open-source models and the majority of proprietary models (e.g., Inflection-2 and Gemini Pro) by at least 9% absolute. This is true even for models that are substantially larger (e.g., Qwen 72B) or have been specifically enhanced through math-focused reinforcement learning (e.g., WizardMath-v1.1 7B). While DeepSeekMath-Instruct rivals the Chinese proprietary models GLM-4 and Baichuan-3 on MATH, it still underperforms GPT-4 and Gemini Ultra.
如表 5 所示，在禁止使用工具的评估设置下，DeepSeekMath-Instruct 7B 展现了强大的分步推理性能。值得注意的是，在竞赛级 MATH 数据集上，我们的模型超过所有开源模型以及大多数闭源模型（如 Inflection-2 和 Gemini Pro）至少 9% 的绝对值。即便对于规模显著更大的模型（如 Qwen 72B）或经过专门数学强化学习增强的模型（如 WizardMath-v1.1 7B）也是如此。虽然 DeepSeekMath-Instruct 在 MATH 上与中国闭源模型 GLM-4 和 Baichuan-3 旗鼓相当，但仍逊于 GPT-4 和 Gemini Ultra。


Under the evaluation setting where models are allowed to integrate natural language reasoning and program-based tool use for problem solving, DeepSeekMath-Instruct 7B approaches an accuracy of 60% on MATH, surpassing all existing open-source models. On the other benchmarks, our model is competitive with DeepSeek-LLM-Chat 67B, the prior state-of-the-art that is 10 times larger.
在允许模型整合自然语言推理和基于程序的工具使用来解决问题的评估设置下，DeepSeekMath-Instruct 7B 在 MATH 上的准确率接近 60%，超越了所有现有开源模型。在其他基准测试中，我们的模型与 DeepSeek-LLM-Chat 67B 相比具有竞争力，后者是此前规模大出 10 倍的最先进模型。


## 4. Reinforcement Learning
## 4. 强化学习


### 4.1. Group Relative Policy Optimization
### 4.1. 群组相对策略优化


Reinforcement learning (RL) has been proven to be effective in further improving the mathematical reasoning ability of LLMs after the Supervised Fine-Tuning (SFT) stage (Luo et al., 2023) Wang et al. 2023b). In this section, we introduce our efficient and effective RL algorithm, Group Relative Policy Optimization (GRPO).
强化学习 (RL) 已被证明在有监督微调 (SFT) 阶段后能进一步有效提升 LLM 的数学推理能力 (Luo et al., 2023; Wang et al. 2023b)。在本节中，我们介绍了一种高效且实用的强化学习算法：群组相对策略优化 (GRPO)。


#### 4.1.1. From PPO to GRPO
#### 4.1.1. 从 PPO 到 GRPO


Proximal Policy Optimization (PPO) (Schulman et al. 2017) is an actor-critic RL algorithm that is widely used in the RL fine-tuning stage of LLMs (Ouyang et al., 2022). In particular, it optimizes LLMs by maximizing the following surrogate objective:
近端策略优化 (PPO) (Schulman et al. 2017) 是一种 actor-critic 强化学习算法，广泛用于 LLM 的强化学习微调阶段 (Ouyang et al., 2022)。具体而言，它通过最大化以下代理目标来优化 LLM：


$$
{\mathcal{J}}_{PPO}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  P\left( Q\right) ,o \sim  {\pi }_{{\theta }_{old}}\left( {O \mid  q}\right) }\right\rbrack  \frac{1}{\left| o\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| o\right| }\min \left\lbrack  {\frac{{\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }{{\pi }_{{\theta }_{old}}\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }{A}_{t},\operatorname{clip}\left( {\frac{{\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }{{\pi }_{{\theta }_{old}}\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) },1 - \varepsilon ,1 + \varepsilon }\right) {A}_{t}}\right\rbrack  , \tag{1}
$$



where ${\pi }_{\theta }$ and ${\pi }_{{\theta }_{\text{ old }}}$ are the current and old policy models,and $q,o$ are questions and outputs sampled from the question dataset and the old policy ${\pi }_{{\theta }_{\text{ old }}}$ ,respectively. $\varepsilon$ is a clipping-related hyper-parameter introduced in PPO for stabilizing training. ${A}_{t}$ is the advantage,which is computed by applying Generalized Advantage Estimation (GAE) (Schulman et al., 2015), based on the rewards $\left\{  {r}_{ \geq  t}\right\}$ and a learned value function ${V}_{\psi }$ . Thus,in PPO,a value function needs to be trained alongside the policy model and to mitigate over-optimization of the reward model, the standard approach is to add a per-token KL penalty from a reference model in the reward at each token (Ouyang et al., 2022), i.e.,
其中 ${\pi }_{\theta }$ 和 ${\pi }_{{\theta }_{\text{ old }}}$ 分别是当前和旧的策略模型，$q,o$ 是分别从问题数据集和旧策略 ${\pi }_{{\theta }_{\text{ old }}}$ 中采样出的问题和输出。$\varepsilon$ 是 PPO 中为稳定训练而引入的剪切相关超参数。${A}_{t}$ 是优势值，它是基于奖励 $\left\{  {r}_{ \geq  t}\right\}$ 和习得的值函数 ${V}_{\psi }$，通过应用广义优势估计 (GAE) (Schulman et al., 2015) 计算得出的。因此，在 PPO 中，需要随策略模型一同训练一个值函数；并且为了缓解奖励模型的过度优化，标准方法是在每个 token 的奖励中添加来自参考模型的逐 token KL 惩罚 (Ouyang et al., 2022)，即：


$$
{r}_{t} = {r}_{\varphi }\left( {q,{o}_{ \leq  t}}\right)  - \beta \log \frac{{\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }{{\pi }_{\text{ ref }}\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }, \tag{2}
$$



---



https://github.com/InternLM/InternLM-Math



---



<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="2">English Benchmarks</td><td colspan="2">Chinese Benchmarks</td></tr><tr><td>GSM8K</td><td>MATH</td><td>MGSM-zh CMATH</td><td></td></tr><tr><td colspan="6">Chain-of-Thought Reasoning</td></tr><tr><td colspan="6">Closed-Source Model</td></tr><tr><td>Gemini Ultra</td><td>-</td><td>94.4%</td><td>53.2%</td><td>-</td><td>-</td></tr><tr><td>GPT-4</td><td>-</td><td>92.0%</td><td>52.9%</td><td>-</td><td>86.0%</td></tr><tr><td>Inflection-2</td><td>-</td><td>81.4%</td><td>34.8%</td><td>-</td><td>-</td></tr><tr><td>GPT-3.5</td><td>-</td><td>80.8%</td><td>34.1%</td><td>-</td><td>73.8%</td></tr><tr><td>Gemini Pro</td><td>-</td><td>86.5%</td><td>32.6%</td><td>-</td><td>-</td></tr><tr><td>Grok-1</td><td>-</td><td>62.9%</td><td>23.9%</td><td>-</td><td>-</td></tr><tr><td>Baichuan-3</td><td>-</td><td>88.2%</td><td>49.2%</td><td>-</td><td>-</td></tr><tr><td>GLM-4</td><td>-</td><td>87.6%</td><td>47.9%</td><td>-</td><td>-</td></tr><tr><td colspan="6">Open-Source Model</td></tr><tr><td>InternLM2-Math</td><td>20B</td><td>82.6%</td><td>37.7%</td><td>-</td><td>-</td></tr><tr><td>Qwen</td><td>72B</td><td>78.9%</td><td>35.2%</td><td>-</td><td>-</td></tr><tr><td>Math-Shepherd-Mistral</td><td>7B</td><td>84.1%</td><td>33.0%</td><td>-</td><td>-</td></tr><tr><td>WizardMath-v1.1</td><td>7B</td><td>83.2%</td><td>33.0%</td><td>-</td><td>-</td></tr><tr><td>DeepSeek-LLM-Chat</td><td>67B</td><td>84.1%</td><td>32.6%</td><td>74.0%</td><td>80.3%</td></tr><tr><td>MetaMath</td><td>70B</td><td>82.3%</td><td>26.6%</td><td>66.4%</td><td>70.9%</td></tr><tr><td>SeaLLM-v2</td><td>7B</td><td>78.2%</td><td>27.5%</td><td>64.8%</td><td>-</td></tr><tr><td>ChatGLM3</td><td>6B</td><td>72.3%</td><td>25.7%</td><td>-</td><td>-</td></tr><tr><td>WizardMath-v1.0</td><td>70B</td><td>81.6%</td><td>22.7%</td><td>64.8%</td><td>65.4%</td></tr><tr><td>DeepSeekMath-Instruct</td><td>7B</td><td>82.9%</td><td>46.8%</td><td>73.2%</td><td>84.6%</td></tr><tr><td>DeepSeekMath-RL</td><td>7B</td><td>88.2%</td><td>51.7%</td><td>79.6%</td><td>88.8%</td></tr><tr><td colspan="6">Tool-Integrated Reasoning</td></tr><tr><td colspan="6">Closed-Source Model</td></tr><tr><td>GPT-4 Code Interpreter</td><td>-</td><td>97.0%</td><td>69.7%</td><td>-</td><td>-</td></tr><tr><td colspan="6">Open-Source Model</td></tr><tr><td>InternLM2-Math</td><td>20B</td><td>80.7%</td><td>54.3%</td><td>-</td><td>-</td></tr><tr><td>DeepSeek-LLM-Chat</td><td>67B</td><td>86.7%</td><td>51.1%</td><td>76.4%</td><td>85.4%</td></tr><tr><td>ToRA</td><td>34B</td><td>80.7%</td><td>50.8%</td><td>41.2%</td><td>53.4%</td></tr><tr><td>MAmmoTH</td><td>70B</td><td>76.9%</td><td>41.8%</td><td>-</td><td>-</td></tr><tr><td>DeepSeekMath-Instruct</td><td>7B</td><td>83.7%</td><td>57.4%</td><td>72.0%</td><td>84.3%</td></tr><tr><td>DeepSeekMath-RL</td><td>7B</td><td>86.7%</td><td>58.8%</td><td>78.4%</td><td>87.6%</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">规模</td><td colspan="2">英文基准</td><td colspan="2">中文基准</td></tr><tr><td>GSM8K</td><td>MATH</td><td>MGSM-zh CMATH</td><td></td></tr><tr><td colspan="6">思维链推理</td></tr><tr><td colspan="6">闭源模型</td></tr><tr><td>Gemini Ultra</td><td>-</td><td>94.4%</td><td>53.2%</td><td>-</td><td>-</td></tr><tr><td>GPT-4</td><td>-</td><td>92.0%</td><td>52.9%</td><td>-</td><td>86.0%</td></tr><tr><td>Inflection-2</td><td>-</td><td>81.4%</td><td>34.8%</td><td>-</td><td>-</td></tr><tr><td>GPT-3.5</td><td>-</td><td>80.8%</td><td>34.1%</td><td>-</td><td>73.8%</td></tr><tr><td>Gemini Pro</td><td>-</td><td>86.5%</td><td>32.6%</td><td>-</td><td>-</td></tr><tr><td>Grok-1</td><td>-</td><td>62.9%</td><td>23.9%</td><td>-</td><td>-</td></tr><tr><td>百川-3</td><td>-</td><td>88.2%</td><td>49.2%</td><td>-</td><td>-</td></tr><tr><td>GLM-4</td><td>-</td><td>87.6%</td><td>47.9%</td><td>-</td><td>-</td></tr><tr><td colspan="6">开源模型</td></tr><tr><td>书生·浦语2-数学</td><td>20B</td><td>82.6%</td><td>37.7%</td><td>-</td><td>-</td></tr><tr><td>通义千问</td><td>72B</td><td>78.9%</td><td>35.2%</td><td>-</td><td>-</td></tr><tr><td>Math-Shepherd-Mistral</td><td>7B</td><td>84.1%</td><td>33.0%</td><td>-</td><td>-</td></tr><tr><td>WizardMath-v1.1</td><td>7B</td><td>83.2%</td><td>33.0%</td><td>-</td><td>-</td></tr><tr><td>DeepSeek-LLM-Chat</td><td>67B</td><td>84.1%</td><td>32.6%</td><td>74.0%</td><td>80.3%</td></tr><tr><td>MetaMath</td><td>70B</td><td>82.3%</td><td>26.6%</td><td>66.4%</td><td>70.9%</td></tr><tr><td>SeaLLM-v2</td><td>7B</td><td>78.2%</td><td>27.5%</td><td>64.8%</td><td>-</td></tr><tr><td>智谱清言3</td><td>6B</td><td>72.3%</td><td>25.7%</td><td>-</td><td>-</td></tr><tr><td>WizardMath-v1.0</td><td>70B</td><td>81.6%</td><td>22.7%</td><td>64.8%</td><td>65.4%</td></tr><tr><td>DeepSeekMath-Instruct</td><td>7B</td><td>82.9%</td><td>46.8%</td><td>73.2%</td><td>84.6%</td></tr><tr><td>DeepSeekMath-RL</td><td>7B</td><td>88.2%</td><td>51.7%</td><td>79.6%</td><td>88.8%</td></tr><tr><td colspan="6">工具集成推理</td></tr><tr><td colspan="6">闭源模型</td></tr><tr><td>GPT-4 代码解释器</td><td>-</td><td>97.0%</td><td>69.7%</td><td>-</td><td>-</td></tr><tr><td colspan="6">开源模型</td></tr><tr><td>书生·浦语2-数学</td><td>20B</td><td>80.7%</td><td>54.3%</td><td>-</td><td>-</td></tr><tr><td>DeepSeek-LLM-Chat</td><td>67B</td><td>86.7%</td><td>51.1%</td><td>76.4%</td><td>85.4%</td></tr><tr><td>ToRA</td><td>34B</td><td>80.7%</td><td>50.8%</td><td>41.2%</td><td>53.4%</td></tr><tr><td>MAmmoTH</td><td>70B</td><td>76.9%</td><td>41.8%</td><td>-</td><td>-</td></tr><tr><td>DeepSeekMath-Instruct</td><td>7B</td><td>83.7%</td><td>57.4%</td><td>72.0%</td><td>84.3%</td></tr><tr><td>DeepSeekMath-RL</td><td>7B</td><td>86.7%</td><td>58.8%</td><td>78.4%</td><td>87.6%</td></tr></tbody></table>


Table 5 | Performance of Open- and Closed-Source models with both Chain-of-Thought and Tool-Integrated Reasoning on English and Chinese Benchmarks. Scores in gray denote majority votes with 32 candidates; The others are Top1 scores. DeepSeekMath-RL 7B beats all open-source models from 7B to 70B, as well as the majority of closed-source models. Although DeepSeekMath-RL 7B is only further trained on chain-of-thought-format instruction tuning data of GSM8K and MATH, it improves over DeepSeekMath-Instruct 7B on all benchmarks.
表 5 | 开源与闭源模型在英文和中文基准测试中结合思维链与工具集成推理的性能。灰色分数表示 32 个候选样本的多重投票结果；其余为 Top1 分数。DeepSeekMath-RL 7B 击败了从 7B 到 70B 的所有开源模型，以及大多数闭源模型。尽管 DeepSeekMath-RL 7B 仅在 GSM8K 和 MATH 的思维链格式指令微调数据上进行了进一步训练，但它在所有基准测试中均优于 DeepSeekMath-Instruct 7B。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_17_11_80589a.jpg"/>



Figure 4 | Demonstration of PPO and our GRPO. GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources.
图 4 | PPO 与我们的 GRPO 演示。GRPO 放弃了价值模型，转而通过组得分估计基准值，显著减少了训练资源。


where ${r}_{\varphi }$ is the reward model, ${\pi }_{\text{ ref }}$ is the reference model,which is usually the initial SFT model, and $\beta$ is the coefficient of the KL penalty.
其中 ${r}_{\varphi }$ 是奖励模型，${\pi }_{\text{ ref }}$ 是参考模型（通常是初始 SFT 模型），$\beta$ 是 KL 惩罚系数。


As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden. Additionally, during RL training, the value function is treated as a baseline in the calculation of the advantage for variance reduction. While in the LLM context, usually only the last token is assigned a reward score by the reward model, which may complicate the training of a value function that is accurate at each token. To address this, as shown in Figure 4, we propose Group Relative Policy Optimization (GRPO), which obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question,as the baseline. More specifically,for each question $q$ ,GRPO samples a group of outputs $\left\{  {{o}_{1},{o}_{2},\cdots ,{o}_{G}}\right\}$ from the old policy ${\pi }_{{\theta }_{\text{ old }}}$ and then optimizes the policy model by maximizing the following objective:
由于 PPO 中使用的价值函数通常是另一个与策略模型规模相当的模型，这带来了巨大的内存和计算负担。此外，在 RL 训练期间，价值函数在计算优势以减少方差时被视为基准值。而在 LLM 场景下，通常只有最后一个 token 被奖励模型分配奖励分数，这可能会使每个 token 都准确的价值函数训练变得复杂。为了解决这一问题，如图 4 所示，我们提出了组相对策略优化 (GRPO)，它消除了 PPO 中对额外价值函数近似的需求，转而使用针对同一问题产生的多个采样输出的平均奖励作为基准值。更具体地，对于每个问题 $q$，GRPO 从旧策略 ${\pi }_{{\theta }_{\text{ old }}}$ 中采样一组输出 $\left\{  {{o}_{1},{o}_{2},\cdots ,{o}_{G}}\right\}$，然后通过最大化以下目标来优化策略模型：


$$
{\mathcal{J}}_{GRPO}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  P\left( Q\right) ,{\left\{  {o}_{i}\right\}  }_{i = 1}^{G} \sim  {\pi }_{{\theta }_{old}}\left( {O \mid  q}\right) }\right\rbrack
$$



$$
\frac{1}{G}\mathop{\sum }\limits_{{i = 1}}^{G}\frac{1}{\left| {o}_{i}\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| {o}_{i}\right| }\left\{  {\min \left\lbrack  {\frac{{\pi }_{\theta }\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) }{{\pi }_{{\theta }_{\text{ old }}}\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) }{\widehat{A}}_{i,t},\operatorname{clip}\left( {\frac{{\pi }_{\theta }\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) }{{\pi }_{{\theta }_{\text{ old }}}\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) },1 - \varepsilon ,1 + \varepsilon }\right) {\widehat{A}}_{i,t}}\right\rbrack   - \beta {\mathbb{D}}_{KL}\left\lbrack  {{\pi }_{\theta }\parallel {\pi }_{\text{ ref }}}\right\rbrack  }\right\}  , \tag{3}
$$



where $\varepsilon$ and $\beta$ are hyper-parameters,and ${\widehat{A}}_{i,t}$ is the advantage calculated based on relative rewards of the outputs inside each group only, which will be detailed in the following subsections. The group relative way that GRPO leverages to calculate the advantages, aligns well with the comparative nature of rewards models, as reward models are typically trained on datasets of comparisons between outputs on the same question. Also note that, instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss,avoiding complicating the calculation of ${\widehat{A}}_{i,t}$ .
其中 $\varepsilon$ 和 $\beta$ 是超参数，${\widehat{A}}_{i,t}$ 是仅基于每组内部输出的相对奖励计算出的优势，详情见下文各小节。GRPO 利用组相对方式计算优势，这与奖励模型的比较性质高度契合，因为奖励模型通常是在针对同一问题的输出比较数据集上训练的。另请注意，GRPO 并非在奖励中加入 KL 惩罚，而是通过直接将训练策略与参考策略之间的 KL 散度添加到损失函数中来进行正则化，从而避免了 ${\widehat{A}}_{i,t}$ 计算的复杂化。


Algorithm 1 Iterative Group Relative Policy Optimization
算法 1 迭代组相对策略优化


---



Input initial policy model ${\pi }_{{\theta }_{\text{ init }}}$ ; reward models ${r}_{\varphi }$ ; task prompts $\mathcal{D}$ ; hyperparameters $\varepsilon ,\beta ,\mu$
输入 初始策略模型 ${\pi }_{{\theta }_{\text{ init }}}$；奖励模型 ${r}_{\varphi }$；任务提示词 $\mathcal{D}$；超参数 $\varepsilon ,\beta ,\mu$


&nbsp;&nbsp;&nbsp;&nbsp;policy model ${\pi }_{\theta } \leftarrow  {\pi }_{{\theta }_{\text{ init }}}$
&nbsp;&nbsp;&nbsp;&nbsp;策略模型 ${\pi }_{\theta } \leftarrow  {\pi }_{{\theta }_{\text{ init }}}$


&nbsp;&nbsp;&nbsp;&nbsp;for iteration $= 1,\ldots ,$ I do
&nbsp;&nbsp;&nbsp;&nbsp;迭代次数 $= 1,\ldots ,$ I 执行：


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;reference model ${\pi }_{\text{ ref }} \leftarrow  {\pi }_{\theta }$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;参考模型 ${\pi }_{\text{ ref }} \leftarrow  {\pi }_{\theta }$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for step $= 1,\ldots ,\mathrm{M}$ do
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;步数 $= 1,\ldots ,\mathrm{M}$ 执行：


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample a batch ${\mathcal{D}}_{b}$ from $\mathcal{D}$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;从 $\mathcal{D}$ 中采样一个批次 ${\mathcal{D}}_{b}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update the old policy model ${\pi }_{{\theta }_{\text{ old }}} \leftarrow  {\pi }_{\theta }$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新旧策略模型 ${\pi }_{{\theta }_{\text{ old }}} \leftarrow  {\pi }_{\theta }$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample $G$ outputs ${\left\{  {o}_{i}\right\}  }_{i = 1}^{G} \sim  {\pi }_{{\theta }_{\text{ old }}}\left( {\cdot  \mid  q}\right)$ for each question $q \in  {\mathcal{D}}_{b}$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为每个问题 $q \in  {\mathcal{D}}_{b}$ 采样 $G$ 个输出 ${\left\{  {o}_{i}\right\}  }_{i = 1}^{G} \sim  {\pi }_{{\theta }_{\text{ old }}}\left( {\cdot  \mid  q}\right)$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute rewards ${\left\{  {r}_{i}\right\}  }_{i = 1}^{G}$ for each sampled output ${o}_{i}$ by running ${r}_{\varphi }$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过运行 ${r}_{\varphi }$ 为每个采样输出 ${o}_{i}$ 计算奖励 ${\left\{  {r}_{i}\right\}  }_{i = 1}^{G}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute ${\widehat{A}}_{i,t}$ for the $t$ -th token of ${o}_{i}$ through group relative advantage estimation.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过组相对优势估计计算 ${o}_{i}$ 的第 $t$ 个 token 的 ${\widehat{A}}_{i,t}$。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for GRPO iteration $= 1,\ldots ,\mu$ do
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对于 GRPO 迭代 $= 1,\ldots ,\mu$ 执行


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update the policy model ${\pi }_{\theta }$ by maximizing the GRPO objective (Equation 21)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过最大化 GRPO 目标函数（公式 21）来更新策略模型 ${\pi }_{\theta }$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update ${r}_{\varphi }$ through continuous training using a replay mechanism.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过使用回放机制的持续训练来更新 ${r}_{\varphi }$。


Output ${\pi }_{\theta }$
输出 ${\pi }_{\theta }$


---



And different from the KL penalty term used in 2), we estimate the KL divergence with the following unbiased estimator (Schulman, 2020):
与 2) 中使用的 KL 惩罚项不同，我们使用以下无偏估计器来估计 KL 散度（Schulman, 2020）：


$$
{\mathbb{D}}_{KL}\left\lbrack  {{\pi }_{\theta }\parallel {\pi }_{\text{ ref }}}\right\rbrack   = \frac{{\pi }_{\text{ ref }}\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) }{{\pi }_{\theta }\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) } - \log \frac{{\pi }_{\text{ ref }}\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) }{{\pi }_{\theta }\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) } - 1, \tag{4}
$$



which is guaranteed to be positive.
这保证了其结果为正。


#### 4.1.2. Outcome Supervision RL with GRPO
#### 4.1.2. 基于 GRPO 的结果监督强化学习


Formally,for each question $q$ ,a group of outputs $\left\{  {{o}_{1},{o}_{2},\cdots ,{o}_{G}}\right\}$ are sampled from the old policy model ${\pi }_{{\theta }_{\text{ old }}}$ . A reward model is then used to score the outputs,yielding $G$ rewards $\mathbf{r} = \left\{  {{r}_{1},{r}_{2},\cdots ,{r}_{G}}\right\}$ correspondingly. Subsequently,these rewards are normalized by subtracting the group average and dividing by the group standard deviation. Outcome supervision provides the normalized reward at the end of each output ${o}_{i}$ and sets the advantages ${\widehat{A}}_{i,t}$ of all tokens in the output as the normalized reward,i.e., ${\widehat{A}}_{i,t} = {\widetilde{r}}_{i} = \frac{{r}_{i} - \operatorname{mean}\left( \mathbf{r}\right) }{\operatorname{std}\left( \mathbf{r}\right) }$ ,and then optimizes the policy by maximizing the objective defined in equation (3).
形式上，对于每个问题 $q$，从旧策略模型 ${\pi }_{{\theta }_{\text{ old }}}$ 中采样一组输出 $\left\{  {{o}_{1},{o}_{2},\cdots ,{o}_{G}}\right\}$。随后使用奖励模型为这些输出评分，产生相应的 $G$ 个奖励 $\mathbf{r} = \left\{  {{r}_{1},{r}_{2},\cdots ,{r}_{G}}\right\}$。接着，通过减去组平均值并除以组标准差来对这些奖励进行归一化。结果监督在每个输出 ${o}_{i}$ 的末尾提供归一化奖励，并将输出中所有 token 的优势值 ${\widehat{A}}_{i,t}$ 设置为该归一化奖励，即 ${\widehat{A}}_{i,t} = {\widetilde{r}}_{i} = \frac{{r}_{i} - \operatorname{mean}\left( \mathbf{r}\right) }{\operatorname{std}\left( \mathbf{r}\right) }$，然后通过最大化公式 (3) 中定义的目标函数来优化策略。


#### 4.1.3. Process Supervision RL with GRPO
#### 4.1.3. 基于 GRPO 的过程监督强化学习


Outcome supervision only provides a reward at the end of each output, which may not be sufficient and efficient to supervise the policy in complex mathematical tasks. Following Wang et al. (2023b), we also explore process supervision, which provides a reward at the end of each reasoning step. Formally,given the question $q$ and $G$ sampled outputs $\left\{  {{o}_{1},{o}_{2},\cdots ,{o}_{G}}\right\}$ ,a process reward model is used to score each step of the outputs, yielding corresponding rewards: $\mathbf{R} = \left\{  {\left\{  {{r}_{1}^{\operatorname{index}\left( 1\right) },\cdots ,{r}_{1}^{\operatorname{index}\left( {K}_{1}\right) }}\right\}  ,\cdots ,\left\{  {{r}_{G}^{\operatorname{index}\left( 1\right) },\cdots ,{r}_{G}^{\operatorname{index}\left( {K}_{G}\right) }}\right\}  }\right\}$ ,where index $\left( j\right)$ is the end token index of the $j$ -th step,and ${K}_{i}$ is the total number of steps in the $i$ -th output. We also normalize these rewards with the average and the standard deviation,i.e., ${\widetilde{r}}_{i}^{\text{ index }\left( j\right) } = \frac{{r}_{i}^{\text{ index }\left( j\right) } - \operatorname{mean}\left( \mathbf{R}\right) }{\operatorname{std}\left( \mathbf{R}\right) }$ . Subsequently, the process supervision calculates the advantage of each token as the sum of the normalized rewards from the following steps,i.e., ${\widehat{A}}_{i,t} = \mathop{\sum }\limits_{{\text{ index }\left( j\right)  \geq  t}}{\widetilde{r}}_{i}^{\text{ index }\left( j\right) }$ ,and then optimizes the policy by maximizing the objective defined in equation (3).
结果监督仅在每个输出结束时提供奖励，这对于监督复杂数学任务中的策略可能不足且低效。参考 Wang 等人 (2023b)，我们也探索了过程监督，即在每个推理步骤结束时提供奖励。形式上，给定问题 $q$ 和 $G$ 个采样输出 $\left\{  {{o}_{1},{o}_{2},\cdots ,{o}_{G}}\right\}$，使用过程奖励模型为输出的每一步评分，产生相应的奖励：$\mathbf{R} = \left\{  {\left\{  {{r}_{1}^{\operatorname{index}\left( 1\right) },\cdots ,{r}_{1}^{\operatorname{index}\left( {K}_{1}\right) }}\right\}  ,\cdots ,\left\{  {{r}_{G}^{\operatorname{index}\left( 1\right) },\cdots ,{r}_{G}^{\operatorname{index}\left( {K}_{G}\right) }}\right\}  }\right\}$，其中索引 $\left( j\right)$ 是第 $j$ 步的结束 token 索引，而 ${K}_{i}$ 是第 $i$ 个输出的总步数。我们同样利用平均值和标准差对这些奖励进行归一化，即 ${\widetilde{r}}_{i}^{\text{ index }\left( j\right) } = \frac{{r}_{i}^{\text{ index }\left( j\right) } - \operatorname{mean}\left( \mathbf{R}\right) }{\operatorname{std}\left( \mathbf{R}\right) }$。随后，过程监督将每个 token 的优势值计算为后续步骤归一化奖励的总和，即 ${\widehat{A}}_{i,t} = \mathop{\sum }\limits_{{\text{ index }\left( j\right)  \geq  t}}{\widetilde{r}}_{i}^{\text{ index }\left( j\right) }$，然后通过最大化公式 (3) 中定义的目标函数来优化策略。


#### 4.1.4. Iterative RL with GRPO
#### 4.1.4. 基于 GRPO 的迭代强化学习


As the reinforcement learning training process progresses, the old reward model may not be sufficient to supervise the current policy model. Therefore, we also explore the iterative RL with GRPO. As shown in Algorithm 1, in iterative GRPO, we generate new training sets for the reward model based on the sampling results from the policy model and continually train the old reward model using a replay mechanism that incorporates 10% of historical data. Then, we set the reference model as the policy model, and continually train the policy model with the new reward model.
随着强化学习训练过程的推进，旧的奖励模型可能不足以监督当前的策略模型。因此，我们还探索了结合GRPO的迭代式强化学习。如算法1所示，在迭代式GRPO中，我们基于策略模型的采样结果为奖励模型生成新的训练集，并利用包含10%历史数据的重放机制持续训练旧的奖励模型。随后，我们将参考模型设为该策略模型，并利用新的奖励模型持续训练策略模型。


### 4.2. Training and Evaluating DeepSeekMath-RL
### 4.2. DeepSeekMath-RL 的训练与评估


We conduct RL based on DeepSeekMath-Instruct 7B. The training data of RL are chain-of-thought-format questions related to GSM8K and MATH from the SFT data, which consists of around 144K questions. We exclude other SFT questions to investigate the impact of RL on benchmarks that lack data throughout the RL phase. We construct the training set of reward models following (Wang et al., 2023b). We train our initial reward model based on the DeepSeekMath-Base 7B with a learning rate of 2e-5. For GRPO, we set the learning rate of the policy model as 1e-6. The KL coefficient is 0.04. For each question, we sample 64 outputs. The max length is set to 1024, and the training batch size is 1024. The policy model only has a single update following each exploration stage. We evaluate DeepSeekMath-RL 7B on benchmarks following DeepSeekMath-Instruct 7B. For DeepSeekMath-RL 7B, GSM8K and MATH with chain-of-thought reasoning can be regarded as in-domain tasks and all the other benchmarks can be regarded as out-of-domain tasks.
我们基于 DeepSeekMath-Instruct 7B 进行强化学习。强化学习的训练数据是来自 SFT 数据中与 GSM8K 和 MATH 相关的思维链格式题目，共计约 14.4 万道。我们排除了其他 SFT 题目，以研究强化学习对整个强化学习阶段都缺乏数据的基准测试的影响。我们遵循 (Wang et al., 2023b) 构建奖励模型的训练集。我们基于 DeepSeekMath-Base 7B 训练初始奖励模型，学习率为 2e-5。对于 GRPO，我们将策略模型的学习率设为 1e-6，KL 系数为 0.04。针对每个问题，我们采样 64 个输出。最大长度设为 1024，训练批大小为 1024。在每个探索阶段后，策略模型仅进行一次更新。我们按照 DeepSeekMath-Instruct 7B 的方式评估 DeepSeekMath-RL 7B。对于 DeepSeekMath-RL 7B，使用思维链推理的 GSM8K 和 MATH 可被视为域内任务，而所有其他基准测试可被视为域外任务。


Table 5 demonstrates the performance of open- and closed-source models with both chain-of-thought and tool-integrated reasoning on English and Chinese benchmarks. We find that: 1) DeepSeekMath-RL 7B attains accuracies of 88.2% and 51.7% on GSM8K and MATH, respectively, utilizing chain-of-thought reasoning. This performance surpasses that of all open-source models in the 7B to 70B range, as well as the majority of closed-source models. 2) Crucially, DeepSeekMath-RL 7B is only trained on chain-of-thought-format instruction tuning data of GSM8K and MATH, starting from DeepSeekMath-Instruct 7B. Despite the constrained scope of its training data, it outperforms DeepSeekMath-Instruct 7B across all evaluation metrics, showcasing the effectiveness of reinforcement learning.
表 5 展示了开源和闭源模型在英文及中文基准测试中，使用思维链和工具集成推理的性能。我们发现：1) DeepSeekMath-RL 7B 利用思维链推理在 GSM8K 和 MATH 上分别达到了 88.2% 和 51.7% 的准确率。这一表现超越了所有 7B 到 70B 范围内的开源模型，以及大多数闭源模型。2) 至关重要的是，DeepSeekMath-RL 7B 仅在来自 DeepSeekMath-Instruct 7B 的 GSM8K 和 MATH 思维链格式指令微调数据上进行训练。尽管训练数据范围受限，它在所有评估指标上均优于 DeepSeekMath-Instruct 7B，展示了强化学习的有效性。


## 5. Discussion
## 5. 讨论


In this section, we will share our findings in pre-training and RL experiments.
在本节中，我们将分享我们在预训练和强化学习实验中的发现。


### 5.1. Lessons Learnt in Pre-Training
### 5.1. 预训练中的经验教训


We first share our experience in pre-training. Unless otherwise specified, we will adhere to the training settings outlined in Section 2.2.1. It is worth noting that, when referring to the DeepSeekMath Corpus in this section, we use an 89B-token dataset from the second iteration of the data collection process.
我们首先分享在预训练方面的经验。除非另有说明，我们将遵循第 2.2.1 节中概述的训练设置。值得注意的是，在本节中提到 DeepSeekMath 语料库时，我们使用的是数据收集过程第二次迭代中包含 89B token 的数据集。


#### 5.1.1. Code Training Benefits Mathematical Reasoning
#### 5.1.1. 代码训练有益于数学推理


A popular yet unverified hypothesis suggests that code training improves reasoning. We attempt to offer a partial response to this, particularly within the mathematical domain: code training
一个流行但尚未证实的假设认为，代码训练可以提高推理能力。我们试图对此提供部分回应，特别是在数学领域：代码训练


<table><tr><td rowspan="2">Training Setting</td><td colspan="3">Training Tokens</td><td colspan="3">w/o Tool Use</td><td colspan="2">w/ Tool Use</td></tr><tr><td>General</td><td>Code</td><td>Math</td><td>GSM8K</td><td>MATH</td><td>CMATH</td><td>GSM8K+Python</td><td>MATH+Python</td></tr><tr><td>No Continual Training</td><td>-</td><td>-</td><td>-</td><td>2.9%</td><td>3.0%</td><td>12.3%</td><td>2.7%</td><td>2.3%</td></tr><tr><td colspan="9">Two-Stage Training</td></tr><tr><td>Stage 1: General Training</td><td>400B</td><td>-</td><td>-</td><td>2.9%</td><td>3.2%</td><td>14.8%</td><td>3.3%</td><td>2.3%</td></tr><tr><td>Stage 2: Math Training</td><td>-</td><td>-</td><td>150B</td><td>19.1%</td><td>14.4%</td><td>37.2%</td><td>14.3%</td><td>6.7%</td></tr><tr><td>Stage 1: Code Training</td><td>-</td><td>400B</td><td>-</td><td>5.9%</td><td>3.6%</td><td>19.9%</td><td>12.4%</td><td>10.0%</td></tr><tr><td>Stage 2: Math Training</td><td>-</td><td>-</td><td>150B</td><td>21.9%</td><td>15.3%</td><td>39.7%</td><td>17.4%</td><td>9.4%</td></tr><tr><td colspan="9">One-Stage Training</td></tr><tr><td>Math Training</td><td>-</td><td>-</td><td>150B</td><td>20.5%</td><td>13.1%</td><td>37.6%</td><td>11.4%</td><td>6.5%</td></tr><tr><td>Code & Math Mixed Training -</td><td></td><td>400B</td><td>150B</td><td>17.6%</td><td>12.1%</td><td>36.3%</td><td>19.7%</td><td>13.5%</td></tr></table>
<table><tbody><tr><td rowspan="2">训练设置</td><td colspan="3">训练 Token 数</td><td colspan="3">不含工具调用</td><td colspan="2">包含工具调用</td></tr><tr><td>通用</td><td>代码</td><td>数学</td><td>GSM8K</td><td>MATH</td><td>CMATH</td><td>GSM8K+Python</td><td>MATH+Python</td></tr><tr><td>无持续训练</td><td>-</td><td>-</td><td>-</td><td>2.9%</td><td>3.0%</td><td>12.3%</td><td>2.7%</td><td>2.3%</td></tr><tr><td colspan="9">两阶段训练</td></tr><tr><td>第一阶段：通用训练</td><td>400B</td><td>-</td><td>-</td><td>2.9%</td><td>3.2%</td><td>14.8%</td><td>3.3%</td><td>2.3%</td></tr><tr><td>第二阶段：数学训练</td><td>-</td><td>-</td><td>150B</td><td>19.1%</td><td>14.4%</td><td>37.2%</td><td>14.3%</td><td>6.7%</td></tr><tr><td>第一阶段：代码训练</td><td>-</td><td>400B</td><td>-</td><td>5.9%</td><td>3.6%</td><td>19.9%</td><td>12.4%</td><td>10.0%</td></tr><tr><td>第二阶段：数学训练</td><td>-</td><td>-</td><td>150B</td><td>21.9%</td><td>15.3%</td><td>39.7%</td><td>17.4%</td><td>9.4%</td></tr><tr><td colspan="9">单阶段训练</td></tr><tr><td>数学训练</td><td>-</td><td>-</td><td>150B</td><td>20.5%</td><td>13.1%</td><td>37.6%</td><td>11.4%</td><td>6.5%</td></tr><tr><td>代码与数学混合训练 -</td><td></td><td>400B</td><td>150B</td><td>17.6%</td><td>12.1%</td><td>36.3%</td><td>19.7%</td><td>13.5%</td></tr></tbody></table>


Table 6 | Investigation of how code affects mathematical reasoning under different training settings. We experiment with DeepSeek-LLM 1.3B, and evaluate its mathematical reasoning performance without and with tool use via few-shot chain-of-thought prompting and few-shot program-of-thought prompting, respectively.
表 6 | 不同训练设置下代码如何影响数学推理的调查。我们以 DeepSeek-LLM 1.3B 进行实验，并分别通过少样本思维链提示和少样本程序思维提示，评估其在不使用和使用工具情况下的数学推理性能。


improves models' ability to do mathematical reasoning both with and without tool use.
提升了模型在有无工具辅助下的数学推理能力。


To study how code training affects mathematical reasoning, we experimented with the following two-stage training and one-stage training settings:
为了研究代码训练如何影响数学推理，我们尝试了以下两阶段训练和一阶段训练设置：


## Two-Stage Training
## 两阶段训练


- Code Training for 400B Tokens $\rightarrow$ Math Training for 150B Tokens: We train DeepSeek-LLM 1.3B for 400B code tokens followed by 150B math tokens;
- 400B Token 代码训练 $\rightarrow$ 150B Token 数学训练：我们对 DeepSeek-LLM 1.3B 先进行 400B 代码 Token 训练，随后进行 150B 数学 Token 训练；


- General Training for 400B Tokens $\rightarrow$ Math Training for 150B Tokens: As a control experiment, we also experiment with general tokens (sampled from a large-scale general corpus created by DeepSeek-AI) instead of code tokens in the first stage of training, in an attempt to investigate the advantages of code tokens over general tokens in improving mathematical reasoning.
- 400B Token 通用训练 $\rightarrow$ 150B Token 数学训练：作为对照实验，我们在第一阶段训练中使用通用 Token（采样自 DeepSeek-AI 构建的大规模通用语料库）代替代码 Token，旨在研究代码 Token 相比通用 Token 在提升数学推理方面的优势。


## One-Stage Training
## 一阶段训练


- Math Training for 150B Tokens: We train DeepSeek-LLM 1.3B for 150B math tokens;
- 150B Token 数学训练：我们对 DeepSeek-LLM 1.3B 进行 150B 数学 Token 训练；


- Training on a mixture of 400B Code Tokens and 150B Math Tokens: Math training following code training degrades coding performance. We investigate whether code tokens, when mixed with math tokens for one-stage training, would still improve mathematical reasoning and also alleviate the problem of catastrophic forgetting.
- 400B 代码 Token 与 150B 数学 Token 混合训练：在代码训练后进行数学训练会降低编程性能。我们研究了在一次性混合训练中，代码 Token 是否仍能提升数学推理，并缓解灾难性遗忘问题。


Results Table 6 and Table 7 demonstrate the downstream performance under different training settings.
结果表 6 和表 7 展示了不同训练设置下的下游任务表现。


Code training benefits program-aided mathematical reasoning, both under the two-stage training and one-stage training settings. As shown in Table 6 under the two-stage training setting, code training alone already significantly enhances the ability to solve GSM8K and MATH problems using Python. Math training in the second stage yields further improvements. Interestingly, under the one-stage training setting, mixing code tokens and math tokens effectively mitigates the issue of catastrophic forgetting that arises from two-stage training, and also synergizes coding (Table 7) and program-aided mathematical reasoning (Table 6).
代码训练在两阶段和一阶段训练设置下均有益于程序辅助数学推理。如表 6 两阶段训练设置所示，仅靠代码训练就已显著增强了使用 Python 解决 GSM8K 和 MATH 问题的能力。第二阶段的数学训练带来了进一步提升。有趣的是，在一阶段训练设置下，混合代码和数学 Token 有效缓解了两阶段训练产生的灾难性遗忘问题，并使代码能力（表 7）与程序辅助数学推理（表 6）产生了协同效应。


<table><tr><td rowspan="2">Training Setting</td><td colspan="3">Training Tokens</td><td rowspan="2">MMLU</td><td rowspan="2">BBH</td><td rowspan="2"></td><td rowspan="2">MBPP (Pass@1)</td></tr><tr><td>General Code Math</td><td></td><td></td></tr><tr><td>No Continual Training</td><td>-</td><td>-</td><td>-</td><td>24.5%</td><td>28.1%</td><td>12.2%</td><td>13.0%</td></tr><tr><td colspan="8">Two-Stage Training</td></tr><tr><td>Stage 1: General Training</td><td>400B</td><td>-</td><td>-</td><td>25.9%</td><td>27.7%</td><td>15.2%</td><td>13.6%</td></tr><tr><td>Stage 2: Math Training</td><td>-</td><td>-</td><td>150B</td><td>33.1%</td><td>32.7%</td><td>12.8%</td><td>13.2%</td></tr><tr><td>Stage 1: Code Training</td><td>-</td><td>400B</td><td>-</td><td>25.0%</td><td>31.5%</td><td>25.0%</td><td>40.0%</td></tr><tr><td>Stage 2: Math Training</td><td>-</td><td>-</td><td>150B</td><td>36.2%</td><td>35.3%</td><td>12.2%</td><td>17.0%</td></tr><tr><td colspan="8">One-Stage Training</td></tr><tr><td>Math Training</td><td>-</td><td>-</td><td>150B</td><td>32.3%</td><td>32.5%</td><td>11.6%</td><td>13.2%</td></tr><tr><td>Code & Math Mixed Training -</td><td></td><td>400B</td><td>150B</td><td>33.5%</td><td>35.6%</td><td>29.3%</td><td>39.4%</td></tr></table>
<table><tbody><tr><td rowspan="2">训练设置</td><td colspan="3">训练 Token 数</td><td rowspan="2">MMLU</td><td rowspan="2">BBH</td><td rowspan="2"></td><td rowspan="2">MBPP (Pass@1)</td></tr><tr><td>通用、代码与数学</td><td></td><td></td></tr><tr><td>无持续训练</td><td>-</td><td>-</td><td>-</td><td>24.5%</td><td>28.1%</td><td>12.2%</td><td>13.0%</td></tr><tr><td colspan="8">两阶段训练</td></tr><tr><td>第一阶段：通用训练</td><td>400B</td><td>-</td><td>-</td><td>25.9%</td><td>27.7%</td><td>15.2%</td><td>13.6%</td></tr><tr><td>第二阶段：数学训练</td><td>-</td><td>-</td><td>150B</td><td>33.1%</td><td>32.7%</td><td>12.8%</td><td>13.2%</td></tr><tr><td>第一阶段：代码训练</td><td>-</td><td>400B</td><td>-</td><td>25.0%</td><td>31.5%</td><td>25.0%</td><td>40.0%</td></tr><tr><td>第二阶段：数学训练</td><td>-</td><td>-</td><td>150B</td><td>36.2%</td><td>35.3%</td><td>12.2%</td><td>17.0%</td></tr><tr><td colspan="8">单阶段训练</td></tr><tr><td>数学训练</td><td>-</td><td>-</td><td>150B</td><td>32.3%</td><td>32.5%</td><td>11.6%</td><td>13.2%</td></tr><tr><td>代码与数学混合训练 -</td><td></td><td>400B</td><td>150B</td><td>33.5%</td><td>35.6%</td><td>29.3%</td><td>39.4%</td></tr></tbody></table>


Table 7 | Investigation of how different settings of code and math training affect model performance of language understanding, reasoning, and coding. We experiment with DeepSeek-LLM 1.3B. We evaluate the models on MMLU and BBH using few-shot chain-of-thought prompting. On HumanEval and MBPP, we conduct zero-shot and few-shot evaluations, respectively.
表 7 | 不同代码与数学训练设置对语言理解、推理及编程性能影响的调查。我们以 DeepSeek-LLM 1.3B 进行实验。在 MMLU 和 BBH 上，我们使用少样本思维链提示进行评估。在 HumanEval 和 MBPP 上，我们分别进行了零样本和少样本评估。


<table><tr><td rowspan="2">Model Size</td><td rowspan="2">ArXiv Corpus</td><td colspan="5">English Benchmarks</td><td colspan="3">Chinese Benchmarks</td></tr><tr><td>GSM8K</td><td>MATH</td><td>OCW</td><td>SAT</td><td>MMLU STEM</td><td>CMATH</td><td>Gaokao MathCloze</td><td>Gaokao MathQA</td></tr><tr><td rowspan="3">DeepSeek-LLM 1.3B</td><td>No Math Training</td><td>2.9%</td><td>3.0%</td><td>2.9%</td><td>15.6%</td><td>19.5%</td><td>12.3%</td><td>0.8%</td><td>17.9%</td></tr><tr><td>MathPile</td><td>2.7%</td><td>3.3%</td><td>2.2%</td><td>12.5%</td><td>15.7%</td><td>1.2%</td><td>0.0%</td><td>2.8%</td></tr><tr><td>ArXiv-RedPajama</td><td>3.3%</td><td>3.4%</td><td>4.0%</td><td>9.4%</td><td>9.0%</td><td>7.4%</td><td>0.8%</td><td>2.3%</td></tr><tr><td rowspan="3">DeepSeek-Coder-Base-v1.5 7B</td><td>No Math Training</td><td>29.0%</td><td>12.5%</td><td>6.6%</td><td>40.6%</td><td>38.1%</td><td>45.9%</td><td>5.9%</td><td>21.1%</td></tr><tr><td>MathPile</td><td>23.6%</td><td>11.5%</td><td>7.0%</td><td>46.9%</td><td>35.8%</td><td>37.9%</td><td>4.2%</td><td>25.6%</td></tr><tr><td>ArXiv-RedPajama</td><td>28.1%</td><td>11.1%</td><td>7.7%</td><td>50.0%</td><td>35.2%</td><td>42.6%</td><td>7.6%</td><td>24.8%</td></tr></table>
<table><tbody><tr><td rowspan="2">模型规模</td><td rowspan="2">ArXiv语料库</td><td colspan="5">英文基准测试</td><td colspan="3">中文基准测试</td></tr><tr><td>GSM8K</td><td>MATH</td><td>OCW</td><td>SAT</td><td>MMLU STEM</td><td>CMATH</td><td>高考数学填空</td><td>高考数学选择</td></tr><tr><td rowspan="3">DeepSeek-LLM 1.3B</td><td>无数学训练</td><td>2.9%</td><td>3.0%</td><td>2.9%</td><td>15.6%</td><td>19.5%</td><td>12.3%</td><td>0.8%</td><td>17.9%</td></tr><tr><td>MathPile</td><td>2.7%</td><td>3.3%</td><td>2.2%</td><td>12.5%</td><td>15.7%</td><td>1.2%</td><td>0.0%</td><td>2.8%</td></tr><tr><td>ArXiv-RedPajama</td><td>3.3%</td><td>3.4%</td><td>4.0%</td><td>9.4%</td><td>9.0%</td><td>7.4%</td><td>0.8%</td><td>2.3%</td></tr><tr><td rowspan="3">DeepSeek-Coder-Base-v1.5 7B</td><td>无数学训练</td><td>29.0%</td><td>12.5%</td><td>6.6%</td><td>40.6%</td><td>38.1%</td><td>45.9%</td><td>5.9%</td><td>21.1%</td></tr><tr><td>MathPile</td><td>23.6%</td><td>11.5%</td><td>7.0%</td><td>46.9%</td><td>35.8%</td><td>37.9%</td><td>4.2%</td><td>25.6%</td></tr><tr><td>ArXiv-RedPajama</td><td>28.1%</td><td>11.1%</td><td>7.7%</td><td>50.0%</td><td>35.2%</td><td>42.6%</td><td>7.6%</td><td>24.8%</td></tr></tbody></table>


Table 8 | Effect of math training on different arXiv datasets. Model performance is evaluated with few-shot chain-of-thought prompting.
表 8 | 数学训练对不同 arXiv 数据集的影响。模型性能通过少样本思维链提示进行评估。


<table><tr><td>ArXiv Corpus</td><td>miniF2F-valid</td><td>miniF2F-test</td></tr><tr><td>No Math Training</td><td>20.1%</td><td>21.7%</td></tr><tr><td>MathPile</td><td>16.8%</td><td>16.4%</td></tr><tr><td>ArXiv-RedPajama</td><td>14.8%</td><td>11.9%</td></tr></table>
<table><tbody><tr><td>ArXiv 语料库</td><td>miniF2F-验证集</td><td>miniF2F-测试集</td></tr><tr><td>无数学训练</td><td>20.1%</td><td>21.7%</td></tr><tr><td>MathPile</td><td>16.8%</td><td>16.4%</td></tr><tr><td>ArXiv-RedPajama</td><td>14.8%</td><td>11.9%</td></tr></tbody></table>


Table 9 | Effect of math training on different arXiv corpora, the base model being DeepSeek-Coder-Base-v1.5 7B. We evaluate informal-to-formal proving in Isabelle.
表 9 | 数学训练对不同 arXiv 语料库的影响，基座模型为 DeepSeek-Coder-Base-v1.5 7B。我们评估了 Isabelle 中的非正式到正式证明转换。


Code training also improves mathematical reasoning without tool use. Under the two-stage training setting, the initial stage of code training already results in moderate enhancements. It also boosts the efficiency of the subsequent math training, eventually leading to the best performance. However, combining code tokens and math tokens for one-stage training compromises mathematical reasoning without tool use. One conjecture is that DeepSeek-LLM 1.3B, due to its limited scale, lacks the capacity to fully assimilate both code and mathematical data simultaneously.
代码训练也能在不使用工具的情况下提高数学推理能力。在两阶段训练设置下，初始阶段的代码训练已经带来了适度提升。它还提高了后续数学训练的效率，最终实现最佳性能。然而，将代码令牌和数学令牌结合进行一阶段训练会损害不使用工具的数学推理能力。一种推测是，由于 DeepSeek-LLM 1.3B 的规模有限，缺乏同时充分吸收代码和数学数据的能力。


#### 5.1.2. ArXiv Papers Seem Ineffective in Improving Mathematical Reasoning
#### 5.1.2. ArXiv 论文在提高数学推理方面似乎无效


ArXiv papers are commonly included as a component of math pre-training data (Azerbayev et al. 2023; Lewkowycz et al. 2022a; Polu and Sutskever, 2020; Wang et al., 2023c). However, detailed analysis regarding their impact on mathematical reasoning has not been extensively conducted. Perhaps counter-intuitively, according to our experiments, arXiv papers seem ineffective in improving mathematical reasoning. We experiment with models of different sizes, including DeepSeek-LLM 1.3B and DeepSeek-Coder-Base-v1.5 7B (Guo et al. 2024), using arXiv corpora that underwent varied processing pipelines:
ArXiv 论文通常被纳入数学预训练数据的一部分 (Azerbayev et al. 2023; Lewkowycz et al. 2022a; Polu and Sutskever, 2020; Wang et al., 2023c)。然而，关于其对数学推理影响的详细分析尚未广泛开展。也许与直觉相反，根据我们的实验，arXiv 论文在提高数学推理方面似乎并不奏效。我们使用经过不同处理流程的 arXiv 语料库，对不同尺寸的模型进行了实验，包括 DeepSeek-LLM 1.3B 和 DeepSeek-Coder-Base-v1.5 7B (Guo et al. 2024)：


- MathPile (Wang et al., 2023c): an 8.9B-token corpus developed with cleaning and filtering heuristic rules, over 85% of which are scientific arXiv papers;
- MathPile (Wang et al., 2023c)：一个包含 8.9B 令牌的语料库，通过清洗和过滤启发式规则开发，其中 85% 以上是科学 arXiv 论文；


- ArXiv-RedPajama (Computer, 2023): the entirety of arXiv LaTeX files with preambles, comments, macros, and bibliographies removed, totaling 28.0B tokens.
- ArXiv-RedPajama (Computer, 2023)：完整的 arXiv LaTeX 文件，移除了序言、注释、宏和参考文献，共计 28.0B 令牌。


In our experiments, we separately train DeepSeek-LLM 1.3B for 150B tokens and DeepSeek-Coder-Base-v1.5 7B for 40B tokens on each arXiv corpus. It seems that arXiv papers are ineffective in improving mathematical reasoning. When trained on a arXiv-only corpus, both models display no notable improvements or even deterioration across various mathematical benchmarks of different complexities employed in this study. These benchmarks include quantitative reasoning datasets like GSM8K and MATH (Table 8), multiple-choice challenges like MMLU-STEM (Table 8), and formal mathematics like miniF2F (Table 9).
在我们的实验中，我们分别在每个 arXiv 语料库上对 DeepSeek-LLM 1.3B 进行了 150B 令牌的训练，对 DeepSeek-Coder-Base-v1.5 7B 进行了 40B 令牌的训练。arXiv 论文在提高数学推理方面似乎无效。当在仅含 arXiv 的语料库上训练时，两个模型在本研究采用的各种不同复杂度的数学基准测试中，均未表现出显著提升，甚至出现退化。这些基准包括定量推理数据集如 GSM8K 和 MATH（表 8）、多项选择挑战如 MMLU-STEM（表 8），以及形式数学如 miniF2F（表 9）。


However, this conclusion has its limitations and should be taken with a grain of salt. We have not yet studied:
然而，这一结论有其局限性，应谨慎对待。我们尚未研究：


- The impact of arXiv tokens on specific math-related tasks not included in this research, such as informalization of theorems which is to convert formal statements or proofs to their informal versions;
- arXiv 令牌对本研究未包含的特定数学相关任务的影响，例如定理的非正式化，即将形式化陈述或证明转换为其非正式版本；


- The effect of arXiv tokens when combined with other types of data;
- arXiv 令牌与其他类型数据结合时的效果；


- Whether the benefits of arXiv papers would manifest themselves at a larger model scale.
- arXiv 论文的益处是否会在更大的模型规模上体现出来。


Thus, further exploration is required, which we leave for future studies.
因此，需要进一步探索，我们将其留给未来的研究。


### 5.2. Insights of Reinforcement Learning
### 5.2. 强化学习的见解


#### 5.2.1. Towards to a Unified Paradigm
#### 5.2.1. 迈向统一范式


In this section, we provide a unified paradigm to analyze different training methods, such as SFT, RFT, DPO, PPO, GRPO, and further conduct experiments to explore the factors of the unified paradigm. Generally,the gradient with respect to the parameter $\theta$ of a training method can be written as:
在本节中，我们提供了一个统一范式来分析不同的训练方法，如 SFT、RFT、DPO、PPO、GRPO，并进一步开展实验以探索统一范式的要素。通常，一种训练方法关于参数 $\theta$ 的梯度可以写为：


$$
{\nabla }_{\theta }{\mathcal{J}}_{\mathcal{A}}\left( \theta \right)  = \mathbb{E}\left\lbrack  \underset{\text{ Data Source }}{\underbrace{\left( {q,o}\right)  \sim  \mathcal{D}}}\right\rbrack  \left( {\frac{1}{\left| o\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| o\right| }\underset{\text{ Gradient Coefficient }}{\underbrace{G{C}_{\mathcal{A}}\left( {q,o,t,{\pi }_{rf}}\right) }}{\nabla }_{\theta }\log {\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }\right) . \tag{5}
$$



There exist three key components: 1) Data Source $\mathcal{D}$ ,which determines the training data; 2) Reward Function ${\pi }_{rf}$ ,which is the source of the training reward signal; 3) Algorithm $\mathcal{A}$ : which processes the training data and the reward signal to the gradient coefficient ${GC}$ that determines the magnitude of the penalty or reinforcement for the data. We analyze several representative methods based on such a unified paradigm:
存在三个关键组件：1) 数据源 $\mathcal{D}$，决定训练数据；2) 奖励函数 ${\pi }_{rf}$，是训练奖励信号的来源；3) 算法 $\mathcal{A}$：将训练数据和奖励信号处理为梯度系数 ${GC}$，决定对数据的惩罚或强化程度。我们基于这种统一范式分析了几种代表性方法：


- Supervised Fine-tuning (SFT): SFT fine-tunes pretrained model on human selected SFT data.
- 有监督微调 (SFT)：SFT 在人工筛选的 SFT 数据上对预训练模型进行微调。


<table><tr><td>Methods</td><td>Data Source</td><td>Reward Function</td><td>Gradient Coefficient</td></tr><tr><td>SFT</td><td>$q,o \sim  {P}_{sft}\left( {Q,O}\right)$</td><td>-</td><td>1</td></tr><tr><td>RFT</td><td>$q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{sft}\left( {O \mid  q}\right)$</td><td>Rule</td><td>Equation 10</td></tr><tr><td>DPO</td><td>$q \sim  {P}_{sft}\left( Q\right) ,{o}^{ + },{o}^{ - } \sim  {\pi }_{sft}\left( {O \mid  q}\right)$</td><td>Rule</td><td>Equation 14</td></tr><tr><td>Online RFT</td><td>$q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{\theta }\left( {O \mid  q}\right)$</td><td>Rule</td><td>Equation 10</td></tr><tr><td>PPO</td><td>$q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{\theta }\left( {O \mid  q}\right)$</td><td>Model</td><td>Equation 18</td></tr><tr><td>GRPO</td><td>$q \sim  {P}_{sft}\left( Q\right) ,{\left\{  {o}_{i}\right\}  }_{i = 1}^{G} \sim  {\pi }_{\theta }\left( {O \mid  q}\right)$</td><td>Model</td><td>Equation 21</td></tr></table>
<table><tbody><tr><td>方法</td><td>数据源</td><td>奖励函数</td><td>梯度系数</td></tr><tr><td>SFT</td><td>$q,o \sim  {P}_{sft}\left( {Q,O}\right)$</td><td>-</td><td>1</td></tr><tr><td>RFT</td><td>$q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{sft}\left( {O \mid  q}\right)$</td><td>规则</td><td>公式 10</td></tr><tr><td>DPO</td><td>$q \sim  {P}_{sft}\left( Q\right) ,{o}^{ + },{o}^{ - } \sim  {\pi }_{sft}\left( {O \mid  q}\right)$</td><td>规则</td><td>公式 14</td></tr><tr><td>在线 RFT</td><td>$q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{\theta }\left( {O \mid  q}\right)$</td><td>规则</td><td>公式 10</td></tr><tr><td>PPO</td><td>$q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{\theta }\left( {O \mid  q}\right)$</td><td>模型</td><td>公式 18</td></tr><tr><td>GRPO</td><td>$q \sim  {P}_{sft}\left( Q\right) ,{\left\{  {o}_{i}\right\}  }_{i = 1}^{G} \sim  {\pi }_{\theta }\left( {O \mid  q}\right)$</td><td>模型</td><td>公式 21</td></tr></tbody></table>


Table 10 | The data source and gradient coefficient of different methods. ${P}_{sft}$ denotes the data distribution of supervised fine-tuning datasets. ${\pi }_{{\theta }_{sft}}$ and ${\pi }_{\theta }$ denote the supervised fine-tuned model and the real-time policy model during the online training process, respectively.
表 10 | 不同方法的数据源与梯度系数。${P}_{sft}$ 表示监督微调数据集的数据分布。${\pi }_{{\theta }_{sft}}$ 和 ${\pi }_{\theta }$ 分别表示监督微调模型和在线训练过程中的实时策略模型。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_17_11_25ed34.jpg"/>



Figure 5 | Performance of the DeepSeekMath-Instruct 1.3B model, which was further trained using various methods, on two benchmarks.
图 5 | DeepSeekMath-Instruct 1.3B 模型在使用各种方法进一步训练后在两个基准测试上的性能。


- Rejection Sampling Fine-tuning (RFT): RFT further fine-tunes the SFT model on the filtered outputs sampled from the SFT model based on SFT questions. RFT filters the outputs based on the correctness of their answers.
- 拒绝采样微调 (RFT)：RFT 在 SFT 模型的基础上，针对 SFT 问题从 SFT 模型采样的筛选输出上进一步微调模型。RFT 根据答案的正确性筛选输出。


- Direct Preference Optimization (DPO): DPO further refines the SFT model by fine-tuning it on augmented outputs sampled from the SFT model, using pair-wise DPO loss.
- 直接偏好优化 (DPO)：DPO 通过使用成对 DPO 损失，在从 SFT 模型采样的增强输出上进行微调，从而进一步优化 SFT 模型。


- Online Rejection Sampling Fine-tuning (Online RFT): Different from RFT, Online RFT initiates the policy model using the SFT model and refines it by fine-tuning with the augmented outputs sampled from the real-time policy model.
- 在线拒绝采样微调 (Online RFT)：与 RFT 不同，Online RFT 使用 SFT 模型初始化策略模型，并通过对从实时策略模型采样的增强输出进行微调来优化模型。


- PPO/GRPO: PPO/GRPO initializes the policy model using the SFT model and reinforces it with the outputs sampled from the real-time policy model.
- PPO/GRPO：PPO/GRPO 使用 SFT 模型初始化策略模型，并使用从实时策略模型采样的输出进行强化。


We summarize the components of these methods in Table 10. Please refer to Appendix A.1 for a more detailed derivation process.
我们在表 10 中总结了这些方法的组成部分。更详细的推导过程请参见附录 A.1。


Observation about Data Source We divide the data source into two categories, online sampling, and offline sampling. Online sampling denotes that the training data is from the exploration results of the real-time training policy model, while offline sampling denotes that the training data is from the sampling results of the initial SFT model. RFT and DPO follow the offline style, while Online RFT and GRPO follow the online style.
关于数据源的观察 我们将数据源分为两类：在线采样和离线采样。在线采样表示训练数据来自实时训练策略模型的探索结果，而离线采样表示训练数据来自初始 SFT 模型的采样结果。RFT 和 DPO 遵循离线模式，而 Online RFT 和 GRPO 遵循在线模式。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_17_11_077e53.jpg"/>



Figure 6 | Performance of iterative reinforcement learning with DeepSeekMath-Instruct 7B on two benchmarks.
图 6 | DeepSeekMath-Instruct 7B 迭代强化学习在两个基准测试上的性能。


As shown in Figure 5, we find that the Online RFT significantly outperforms RFT on two benchmarks. Specifically, Online RFT is comparable to RFT in the early stage of training but gains an absolute advantage in the later stage, demonstrating the superiority of online training. This is intuitive, as in the initial stage, the actor and the SFT model exhibit close resemblance, with the sampled data revealing only minor differences. In the later stage, however, the data sampled from the actor will exhibit more significant differences, and real-time data sampling will offer greater advantages.
如图 5 所示，我们发现 Online RFT 在两个基准测试上均显著优于 RFT。具体而言，Online RFT 在训练初期与 RFT 相当，但在后期获得绝对优势，证明了在线训练的优越性。这很直观，因为在初始阶段，演员模型与 SFT 模型非常相似，采样数据的差异很小。然而在后期，从演员模型采样的数据将表现出更显著的差异，实时数据采样将提供更大的优势。


Observation about Gradient Coefficient The algorithm processes the reward signal to the gradient coefficient to update the model parameter. We divide the reward function as 'Rule' and 'Model' in our experiments. Rule refers to judging the quality of a response based on the correctness of the answer, and Model denotes that we train a reward model to score each response. The training data of the reward model is based on the rule judgment. Equations 10 and 21 highlight a key difference between GRPO and Online RFT: GRPO uniquely adjusts its gradient coefficient based on the reward value provided by the reward model. This allows for differential reinforcement and penalization of responses according to their varying magnitudes. In contrast, Online RFT lacks this feature; it does not penalize incorrect responses and uniformly reinforces all responses with correct answers at the same level of intensity.
关于梯度系数的观察 算法处理奖励信号并将其转化为梯度系数以更新模型参数。在实验中，我们将奖励函数分为“规则”和“模型”。规则是指根据答案的正确性判断回答的质量，模型是指我们训练一个奖励模型为每个回答评分。奖励模型的训练数据基于规则判断。等式 10 和 21 强调了 GRPO 和 Online RFT 之间的一个关键区别：GRPO 独特地根据奖励模型提供的奖励值调整其梯度系数。这允许根据回答的不同量级进行差异化的强化和惩罚。相比之下，Online RFT 缺乏这一特性；它不惩罚错误的回答，并以相同的强度均匀地强化所有答案正确的回答。


As demonstrated in Figure 5, GRPO surpasses online RFT, thereby highlighting the efficiency of altering positive and negative gradient coefficients. In addition, GRPO+PS shows superior performance compared to GRPO+OS, indicating the benefits of using fine-grained, step-aware gradient coefficients. Furthermore, we explore the iterative RL, in our experiments, we conduct two rounds of iteration. As shown in Figure 6, we notice that the iterative RL significantly improves the performance, especially at the first iteration.
如图 5 所示，GRPO 超过了 Online RFT，从而突显了改变正负梯度系数的效率。此外，GRPO+PS 表现出优于 GRPO+OS 的性能，表明使用细粒度、步感知梯度系数的好处。此外，我们探索了迭代强化学习，在实验中，我们进行了两轮迭代。如图 6 所示，我们注意到迭代强化学习显著提高了性能，尤其是在第一次迭代时。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_17_11_6a15cb.jpg"/>



Figure 7 | The Maj@K and Pass@K of SFT and RL DeepSeekMath 7B on GSM8K and MATH (temperature 0.7). It was noted that RL enhances Maj@K but not Pass@K.
图 7 | SFT 和 RL DeepSeekMath 7B 在 GSM8K 和 MATH 上的 Maj@K 和 Pass@K（温度 0.7）。注意到 RL 增强了 Maj@K，但没有增强 Pass@K。


##### 5.2.2.Why RL Works?
##### 5.2.2.为什么强化学习有效？


In this paper, we conduct reinforcement learning based on a subset of instruction tuning data, and it achieves significant performance enhancement upon the instruction tuning model. To further explain why reinforcement learning works. We evaluate the Pass@K and Maj@K accuracy of the Instruct and RL models on two benchmarks. As shown in Figure 7, RL enhances Maj@K's performance but not Pass@K.These findings indicate that RL enhances the model's overall performance by rendering the output distribution more robust, in other words, it seems that the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities. Similarly, (Wang et al., 2023a) identified a misalignment problem in reasoning tasks within the SFT model, showing that the reasoning performance of SFT models can be improved through a series of preference alignment strategies (Song et al., 2023; Wang et al., 2023a; Yuan et al., 2023b).
在本文中，我们基于指令微调数据的子集进行强化学习，其在指令微调模型的基础上实现了显著的性能提升。为进一步解释强化学习为何有效，我们在两个基准测试上评估了 Instruct 和 RL 模型的 Pass@K 与 Maj@K 准确率。如图 7 所示，RL 提升了 Maj@K 的表现，但未提升 Pass@K。这些发现表明，RL 通过使输出分布更具鲁棒性来增强模型的整体性能；换言之，这种提升似乎归功于增加了 TopK 中的正确响应，而非增强了基础能力。同样地，(Wang et al., 2023a) 指出了 SFT 模型在推理任务中的对齐失调问题，并表明通过一系列偏好对齐策略 (Song et al., 2023; Wang et al., 2023a; Yuan et al., 2023b) 可以提高 SFT 模型的推理性能。


##### 5.2.3.How to Achieve More Effective RL?
##### 5.2.3. 如何实现更有效的 RL？


We demonstrate RL works pretty well in mathematical reasoning tasks. We also provide a unified paradigm to understand different representative training methods. Within this paradigm, all methods are conceptualized as either direct or simplified RL techniques. As summarized in Equation 5, there exist three key components: Data Source, Algorithm, and Reward Function. We provide some potential future directions about the three components.
我们证明了 RL 在数学推理任务中表现出色。我们还提供了一个统一范式来理解不同的代表性训练方法。在此范式下，所有方法都被概念化为直接或简化的 RL 技术。如等式 5 所示，存在三个关键组件：数据源、算法和奖励函数。我们就这三个组件提供了一些潜在的未来方向。


Data Source Data source is the raw material of all training methods. In the context of RL, we specifically refer to the data source as the unlabeled questions with the outputs sampled from the policy model. In this paper, we only use the questions from the instruction tuning stage and a naive nucleus sampling to sample outputs. We think this is a potential reason that our RL pipeline only improves the Maj@K performance. In the future, we will explore our RL pipeline on out-of-distribution question prompts, in conjunction with advanced sampling (decoding) strategies, like those based on tree-search methods (Yao et al., 2023). Also, the efficient inference techniques (Kwon et al., 2023; Leviathan et al., 2023; Xia et al., 2023, 2024), which determines the exploration efficiency of policy models, also play an exceedingly important role.
数据源：数据源是所有训练方法的原材料。在 RL 的语境下，我们专门将数据源定义为带有从策略模型采样输出的无标签问题。在本文中，我们仅使用了指令微调阶段的问题和朴素的核采样来采样输出。我们认为这是我们的 RL 流程仅提升 Maj@K 性能的潜在原因。未来，我们将结合先进的采样（解码）策略（如基于树搜索的方法 (Yao et al., 2023)），在分布外的问题提示词上探索我们的 RL 流程。此外，决定策略模型探索效率的高效推理技术 (Kwon et al., 2023; Leviathan et al., 2023; Xia et al., 2023, 2024) 也起着极其重要的作用。


Algorithms Algorithms process the data and reward signal to the gradient coefficient to update the model parameter. Based on Equation 5, to some extent, all methods now fully TRUST the signal of the reward function to increase or decrease the conditional probability of a certain token. However, it is impossible to ensure the reward signal is always reliable, especially in extremely complex tasks. For example, even the PRM800K datasets (Lightman et al., 2023), which have been carefully annotated by well-trained annotators, still contain approximately 20% of incorrectly annotations 7 To this end, we will explore the reinforcement learning algorithm that is robust against noisy reward signals. We believe such WEAK-TO-STRONG (Burns et al., 2023) alignment methods will bring a fundamental change to the learning algorithms.
算法：算法处理数据和奖励信号，并将其转化为梯度系数以更新模型参数。基于等式 5，在某种程度上，目前所有方法都完全“信任”奖励函数的信号，以增加或减少特定 Token 的条件概率。然而，无法确保奖励信号始终可靠，尤其是在极其复杂的任务中。例如，即使是经过训练有素的标注员精心标注的 PRM800K 数据集 (Lightman et al., 2023)，仍包含约 20% 的错误标注。为此，我们将探索对噪声奖励信号具有鲁棒性的强化学习算法。我们相信这种“弱到强”(Burns et al., 2023) 的对齐方法将为学习算法带来根本性的变革。


Reward Function Reward function is the source of the training signal. In RL, the reward function is usually the neural reward model. We think there exist three important directions for reward models: 1) How to enhance the generalization ability of the reward model. The reward model must be effectively generalized to handle out-of-distribution questions and advanced decoding outputs; otherwise, reinforcement learning may merely stabilize the distribution of LLMs rather than improve their fundamental capabilities; 2) How to reflect the uncertainty of reward model. The uncertainty could potentially act as a linking bridge between the weak reward model and the weak-to-strong learning algorithms; 3) How to efficiently build high-quality process reward models that can provide fine-grained training signals for the reasoning process (Lightman et al., 2023; Wang et al., 2023b).
奖励函数：奖励函数是训练信号的来源。在 RL 中，奖励函数通常是神经奖励模型。我们认为奖励模型存在三个重要方向：1) 如何增强奖励模型的泛化能力。奖励模型必须能有效泛化以处理分布外问题和先进的解码输出；否则，强化学习可能仅是稳定了 LLM 的分布，而非提升其基础能力；2) 如何反映奖励模型的不确定性。这种不确定性可能成为弱奖励模型与“弱到强”学习算法之间的纽带；3) 如何高效构建高质量的过程奖励模型，从而为推理过程提供细粒度的训练信号 (Lightman et al., 2023; Wang et al., 2023b)。


## 6. Conclusion, Limitation, and Future Work
## 6. 结论、局限性与未来工作


We present DeepSeekMath, which outperforms all open-source models on the competition-level MATH benchmark and approaches the performance of closed models. DeepSeekMath is initialized with DeepSeek-Coder-v1.5 7B and undergoes continual training for 500B tokens, with a significant component of the training data being 120B math tokens sourced from Common Crawl. Our extensive ablation study shows web pages offer significant potential for high-quality mathematical data, while arXiv may not as beneficial as we expected. We introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), which can notably improve mathematical reasoning capabilities with less memory consumption. The experiment results show that GRPO is effective even if DeepSeekMath-Instruct 7B has reached a high score on benchmarks. We also provide a unified paradigm to understand a series of methods and summarize several potential directions for more effective reinforcement learning.
我们推出了 DeepSeekMath，它在竞赛级 MATH 基准测试中超越了所有开源模型，并接近闭源模型的性能。DeepSeekMath 以 DeepSeek-Coder-v1.5 7B 为基础进行初始化，并经过了 500B Token 的持续训练，其中重要的训练数据组成部分是源自 Common Crawl 的 120B 数学 Token。我们广泛的消融研究表明，网页对于高质量数学数据具有巨大潜力，而 arXiv 的贡献可能不如我们预期的那样大。我们引入了群组相对策略优化 (GRPO)，这是近端策略优化 (PPO) 的一种变体，它能以更少的显存消耗显著提升数学推理能力。实验结果表明，即使 DeepSeekMath-Instruct 7B 在基准测试上已达到高分，GRPO 依然有效。我们还提供了一个统一范式来理解一系列方法，并总结了实现更有效强化学习的几个潜在方向。


Although DeepSeekMath achieves impressive scores on quantitative reasoning benchmarks, its capability on geometry and theorem-proof are relatively weaker than closed models. For instance, in our dry run, the model cannot handle problems related to triangles and ellipses, which may indicate data selection bias in pre-training and fine-tuning. In addition, restricted by the model scale, DeepSeekMath is worse than GPT-4 on few-shot capability. GPT-4 could improve its performance with few-shot inputs, while DeepSeekMath shows similar performance in zero-shot and few-shot evaluation. In the future, we will further improve our engineered data selection pipeline to construct more high-quality pre-trained corpus. In addition, we will explore the potential directions (Section 5.2.3) for more effective reinforcement learning of LLMs.
尽管 DeepSeekMath 在数量推理基准测试中取得了令人瞩目的成绩，但其在几何和定理证明方面的能力相对弱于闭源模型。例如，在我们的初步测试中，该模型无法处理与三角形和椭圆相关的问题，这可能表明预训练和微调中存在数据选择偏差。此外，受限于模型规模，DeepSeekMath 在少样本能力上逊于 GPT-4。GPT-4 可以通过少样本输入提升性能，而 DeepSeekMath 在零样本和少样本评估中表现相近。未来，我们将进一步改进数据选择工程流水线，以构建更高质量的预训练语料库。此外，我们还将探索更有效的 LLM 强化学习潜在方向（第 5.2.3 节）。


---



7https://github.com/openai/prm800k/issues/12#issuecomment-1728491852
7https://github.com/openai/prm800k/issues/12#issuecomment-1728491852


---



## References
## 参考文献


R. Anil, S. Borgeaud, Y. Wu, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, D. Silver, S. Petrov, M. Johnson, I. Antonoglou, J. Schrittwieser, A. Glaese, J. Chen, E. Pitler, T. P. Lillicrap, A. Lazaridou, O. Firat, J. Molloy, M. Isard, P. R. Barham, T. Hennigan, B. Lee, F. Viola, M. Reynolds, Y. Xu, R. Doherty, E. Collins, C. Meyer, E. Rutherford, E. Moreira, K. Ayoub, M. Goel, G. Tucker, E. Piqueras, M. Krikun, I. Barr, N. Savinov, I. Danihelka, B. Roelofs, A. White, A. Andreassen, T. von Glehn, L. Yagati, M. Kazemi, L. Gonzalez, M. Khalman, J. Sygnowski, and et al. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023. doi: 10.48550/ARXIV.2312.11805. URL https: //doi.org/10.48550/arXiv.2312.11805.
R. Anil, S. Borgeaud, Y. Wu, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, D. Silver, S. Petrov, M. Johnson, I. Antonoglou, J. Schrittwieser, A. Glaese, J. Chen, E. Pitler, T. P. Lillicrap, A. Lazaridou, O. Firat, J. Molloy, M. Isard, P. R. Barham, T. Hennigan, B. Lee, F. Viola, M. Reynolds, Y. Xu, R. Doherty, E. Collins, C. Meyer, E. Rutherford, E. Moreira, K. Ayoub, M. Goel, G. Tucker, E. Piqueras, M. Krikun, I. Barr, N. Savinov, I. Danihelka, B. Roelofs, A. White, A. Andreassen, T. von Glehn, L. Yagati, M. Kazemi, L. Gonzalez, M. Khalman, J. Sygnowski, and et al. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023. doi: 10.48550/ARXIV.2312.11805. URL https: //doi.org/10.48550/arXiv.2312.11805.


J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.
J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.


Z. Azerbayev, H. Schoelkopf, K. Paster, M. D. Santos, S. McAleer, A. Q. Jiang, J. Deng, S. Bider-man, and S. Welleck. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631, 2023.
Z. Azerbayev, H. Schoelkopf, K. Paster, M. D. Santos, S. McAleer, A. Q. Jiang, J. Deng, S. Bider-man, and S. Welleck. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631, 2023.


J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.


C. Burns, P. Izmailov, J. H. Kirchner, B. Baker, L. Gao, L. Aschenbrenner, Y. Chen, A. Ecoffet, M. Joglekar, J. Leike, et al. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. arXiv preprint arXiv:2312.09390, 2023.
C. Burns, P. Izmailov, J. H. Kirchner, B. Baker, L. Gao, L. Aschenbrenner, Y. Chen, A. Ecoffet, M. Joglekar, J. Leike, et al. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. arXiv preprint arXiv:2312.09390, 2023.


ChatGLM3 Team. Chatglm3 series: Open bilingual chat llms, 2023. URL https://github.c om/THUDM/ChatGLM3
ChatGLM3 Team. Chatglm3 series: Open bilingual chat llms, 2023. URL https://github.c om/THUDM/ChatGLM3


M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374, 2021. URL https://arxiv.org/abs/2107.03374
M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. 评估在代码上训练的大型语言模型. CoRR, abs/2107.03374, 2021. URL https://arxiv.org/abs/2107.03374


W. Chen, X. Ma, X. Wang, and W. W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. CoRR, abs/2211.12588, 2022. doi: 10.48550/ARXIV.2211.12588.URL.https://doi.org/10.48550/arXiv.2211.12588.
W. Chen, X. Ma, X. Wang, and W. W. Cohen. 思维程序提示：在数值推理任务中将计算与推理分离. CoRR, abs/2211.12588, 2022. doi: 10.48550/ARXIV.2211.12588.URL.https://doi.org/10.48550/arXiv.2211.12588.


K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. 训练验证器以解决数学应用题. arXiv preprint arXiv:2110.14168, 2021.


T. Computer. Redpajama: an open dataset for training large language models, Oct. 2023. URL https://github.com/togethercomputer/RedPajama-Data
T. Computer. Redpajama：一个用于训练大型语言模型的开源数据集，2023年10月. URL https://github.com/togethercomputer/RedPajama-Data


DeepSeek-AI. Deepseek LLM: scaling open-source language models with longtermism. CoRR, abs/2401.02954, 2024. doi: 10.48550/ARXIV.2401.02954. URL https://doi.org/10.485 50/arXiv.2401.02954
DeepSeek-AI. Deepseek LLM：以长期主义扩展开源语言模型. CoRR, abs/2401.02954, 2024. doi: 10.48550/ARXIV.2401.02954. URL https://doi.org/10.485 50/arXiv.2401.02954


Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang. Glm: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320-335, 2022.
Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang. Glm：通过自回归空白填充进行通用语言模型预训练. 见：第60届计算语言学协会年会论文集（第一卷：长篇论文），第320-335页，2022.


L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig. PAL: program-aided language models. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 10764-10799. PMLR, 2023. URL https://proceedings.mlr.press/v202/gao23f html
L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig. PAL：程序辅助语言模型. 见：A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett（编），国际机器学习大会（ICML 2023），2023年7月23-29日，美国夏威夷檀香山，机器学习研究论文集第202卷，第10764-10799页. PMLR, 2023. URL https://proceedings.mlr.press/v202/gao23f html


Z. Gou, Z. Shao, Y. Gong, Y. Shen, Y. Yang, M. Huang, N. Duan, and W. Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. CoRR, abs/2309.17452, 2023. doi: 10.48550/ARXIV.2309.17452. URL https://doi.org/10.48550/arXiv.2309.1745 2
Z. Gou, Z. Shao, Y. Gong, Y. Shen, Y. Yang, M. Huang, N. Duan, and W. Chen. Tora：一种用于数学问题解决的工具集成推理智能体. CoRR, abs/2309.17452, 2023. doi: 10.48550/ARXIV.2309.17452. URL https://doi.org/10.48550/arXiv.2309.1745 2


D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. K. Li, F. Luo, Y. Xiong, and W. Liang. Deepseek-coder: When the large language model meets programming - the rise of code intelligence, 2024.
D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. K. Li, F. Luo, Y. Xiong, and W. Liang. Deepseek-coder: 当大语言模型遇上编程——代码智能的崛起, 2024.


D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. 衡量大规模多任务语言理解. arXiv preprint arXiv:2009.03300, 2020.


D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. 利用 MATH 数据集衡量数学问题解决能力. arXiv preprint arXiv:2103.03874, 2021.


High-flyer. Hai-llm: 高效且轻量的大模型训练工具, 2023. URL https://www.high-flyer.c n/en/blog/hai-llm
幻方量化. Hai-llm: 高效且轻量的大模型训练工具, 2023. URL https://www.high-flyer.cn/en/blog/hai-llm


Inflection AI. Inflection-2, 2023. URL https://inflection.ai/inflection-2
Inflection AI. Inflection-2, 2023. URL https://inflection.ai/inflection-2


A. Q. Jiang, S. Welleck, J. P. Zhou, W. Li, J. Liu, M. Jamnik, T. Lacroix, Y. Wu, and G. Lample. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. arXiv preprint arXiv:2210.12283, 2022.
A. Q. Jiang, S. Welleck, J. P. Zhou, W. Li, J. Liu, M. Jamnik, T. Lacroix, Y. Wu, and G. Lample. 草拟、构思与证明：以非正式证明引导形式化定理证明器. arXiv preprint arXiv:2210.12283, 2022.


A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. I. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. I. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.


A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, and T. Mikolov. Fasttext. zip: Compressing text classification models. arXiv preprint arXiv:1612.03651, 2016.
A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, and T. Mikolov. Fasttext. zip: 压缩文本分类模型. arXiv preprint arXiv:1612.03651, 2016.


W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica. 基于 PagedAttention 的大语言模型服务高效显存管理. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.


Y. Leviathan, M. Kalman, and Y. Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274-19286. PMLR, 2023.
Y. Leviathan, M. Kalman, and Y. Matias. 通过推测解码实现 Transformer 的快速推理. In International Conference on Machine Learning, pages 19274-19286. PMLR, 2023.


A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843-3857, 2022a.
A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, et al. 利用语言模型解决定量推理问题. Advances in Neural Information Processing Systems, 35:3843-3857, 2022a.


A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, Y. Wu, B. Neyshabur, G. Gur-Ari, and V. Misra. Solving quantitative reasoning problems with language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022b. URL http://papers.nips cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstr act-Conference.html
A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, Y. Wu, B. Neyshabur, G. Gur-Ari, and V. Misra. 利用语言模型解决定量推理问题. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022b. URL http://papers.nips.cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html


H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe. Let's verify step by step. arXiv preprint arXiv:2305.20050, 2023.
H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, 和 K. Cobbe. 让我们一步步验证。arXiv 预印本 arXiv:2305.20050, 2023。


I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
I. Loshchilov 和 F. Hutter. 解耦权重衰减正则化。arXiv 预印本 arXiv:1711.05101, 2017。


H. Luo, Q. Sun, C. Xu, P. Zhao, J. Lou, C. Tao, X. Geng, Q. Lin, S. Chen, and D. Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583, 2023.
H. Luo, Q. Sun, C. Xu, P. Zhao, J. Lou, C. Tao, X. Geng, Q. Lin, S. Chen, 和 D. Zhang. Wizardmath：通过强化进化指令增强大语言模型的数学推理能力。arXiv 预印本 arXiv:2308.09583, 2023。


S. Mishra, M. Finlayson, P. Lu, L. Tang, S. Welleck, C. Baral, T. Rajpurohit, O. Tafjord, A. Sab-harwal, P. Clark, and A. Kalyan. LILA: A unified benchmark for mathematical reasoning. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 5807-5832. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.392. URL https://doi.org/10.18653/v1/ 2022.emnlp-main.392
S. Mishra, M. Finlayson, P. Lu, L. Tang, S. Welleck, C. Baral, T. Rajpurohit, O. Tafjord, A. Sab-harwal, P. Clark, 和 A. Kalyan. LILA：统一的数学推理基准。收录于 Y. Goldberg, Z. Kozareva, 和 Y. Zhang 编，《2022年自然语言处理经验方法会议论文集 (EMNLP 2022)》，阿联酋阿布扎比，2022年12月7-11日，5807-5832页。计算语言学协会，2022。doi: 10.18653/V1/2022.EMNLP-MAIN.392. URL https://doi.org/10.18653/v1/ 2022.emnlp-main.392


X. Nguyen, W. Zhang, X. Li, M. M. Aljunied, Q. Tan, L. Cheng, G. Chen, Y. Deng, S. Yang, C. Liu, H. Zhang, and L. Bing. Seallms - large language models for southeast asia. CoRR, abs/2312.00738, 2023. doi: 10.48550/ARXIV.2312.00738. URL https://doi.org/10.485 50/arXiv.2312.00738.
X. Nguyen, W. Zhang, X. Li, M. M. Aljunied, Q. Tan, L. Cheng, G. Chen, Y. Deng, S. Yang, C. Liu, H. Zhang, 和 L. Bing. Seallms - 东南亚大语言模型。CoRR, abs/2312.00738, 2023. doi: 10.48550/ARXIV.2312.00738. URL https://doi.org/10.485 50/arXiv.2312.00738。


OpenAI. GPT4 technical report. arXiv preprint arXiv:2303.08774, 2023.
OpenAI. GPT4 技术报告。arXiv 预印本 arXiv:2303.08774, 2023。


L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744, 2022.
L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, 等。训练语言模型以通过人类反馈遵循指令。神经信息处理系统进展，35:27730-27744, 2022。


K. Paster, M. D. Santos, Z. Azerbayev, and J. Ba. Openwebmath: An open dataset of high-quality mathematical web text. CoRR, abs/2310.06786, 2023. doi: 10.48550/ARXIV.2310.06786. URL https://doi.org/10.48550/arXiv.2310.06786.
K. Paster, M. D. Santos, Z. Azerbayev, 和 J. Ba. Openwebmath：高质量数学网络文本的开源数据集。CoRR, abs/2310.06786, 2023. doi: 10.48550/ARXIV.2310.06786. URL https://doi.org/10.48550/arXiv.2310.06786。


L. C. Paulson. Three years of experience with sledgehammer, a practical link between automatic and interactive theorem provers. In R. A. Schmidt, S. Schulz, and B. Konev, editors, Proceedings of the 2nd Workshop on Practical Aspects of Automated Reasoning, PAAR-2010, Edinburgh, Scotland, UK, July 14, 2010, volume 9 of EPiC Series in Computing, pages 1-10. EasyChair, 2010. doi: 10.29007/TNFD. URL https://doi.org/10.29007/tnfd.
L. C. Paulson. Sledgehammer 三年经验：自动定理证明器与交互式定理证明器之间的实用纽带。收录于 R. A. Schmidt, S. Schulz, 和 B. Konev 编，《第二届自动推理实用方面研讨会论文集 (PAAR-2010)》，英国苏格兰爱丁堡，2010年7月14日，EPiC 计算系列第 9 卷，1-10页。EasyChair, 2010. doi: 10.29007/TNFD. URL https://doi.org/10.29007/tnfd。


S. Polu and I. Sutskever. Generative language modeling for automated theorem proving. CoRR, abs/2009.03393, 2020. URL https://arxiv.org/abs/2009.03393.
S. Polu 和 I. Sutskever. 自动定理证明的生成式语言建模。CoRR, abs/2009.03393, 2020. URL https://arxiv.org/abs/2009.03393。


R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. 2023.
R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, 和 C. Finn. 直接偏好优化：你的语言模型其实是一个奖励模型。2023。


J. Schulman. Approximating kl divergence, 2020. URL http://joschu.net/blog/kl-app rox.html
J. Schulman. 近似 KL 散度，2020. URL http://joschu.net/blog/kl-app rox.html


J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.
J. Schulman, P. Moritz, S. Levine, M. Jordan, 和 P. Abbeel. 使用广义优势估计的高维连续控制。arXiv 预印本 arXiv:1506.02438, 2015。


J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. 近端策略优化算法。arXiv preprint arXiv:1707.06347, 2017.


F. Shi, M. Suzgun, M. Freitag, X. Wang, S. Srivats, S. Vosoughi, H. W. Chung, Y. Tay, S. Ruder, D. Zhou, D. Das, and J. Wei. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id= fR3wGCk-IXp
F. Shi, M. Suzgun, M. Freitag, X. Wang, S. Srivats, S. Vosoughi, H. W. Chung, Y. Tay, S. Ruder, D. Zhou, D. Das, and J. Wei. 语言模型是多语言思维链推理器。第十一届国际学习表征会议（ICLR 2023），卢旺达基加利，2023年5月1-5日。OpenReview.net, 2023. URL https://openreview.net/pdf?id= fR3wGCk-IXp


F. Song, B. Yu, M. Li, H. Yu, F. Huang, Y. Li, and H. Wang. Preference ranking optimization for human alignment. arXiv preprint arXiv:2306.17492, 2023.
F. Song, B. Yu, M. Li, H. Yu, F. Huang, Y. Li, and H. Wang. 面向人类对齐的偏好排名优化。arXiv preprint arXiv:2306.17492, 2023.


M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.
M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, et al. 极具挑战性的 BIG-bench 任务及思维链能否解决它们。arXiv preprint arXiv:2210.09261, 2022.


T. Tao. Embracing change and resetting expectations, 2023. URL https://unlocked.micro soft.com/ai-anthology/terence-tao/
T. Tao. 拥抱变革并重置预期, 2023. URL https://unlocked.micro soft.com/ai-anthology/terence-tao/


H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288, 2023. doi: 10.48550/arXiv.2307.09288. URL https://doi.org/10.48550/arXiv.2307 09288
H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: 开放式基础与微调聊天模型。CoRR, abs/2307.09288, 2023. doi: 10.48550/arXiv.2307.09288. URL https://doi.org/10.48550/arXiv.2307 09288


T. H. Trinh, Y. Wu, Q. V. Le, H. He, and T. Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476-482, 2024.
T. H. Trinh, Y. Wu, Q. V. Le, H. He, and T. Luong. 无需人类演示求解奥数几何题。Nature, 625(7995):476-482, 2024.


P. Wang, L. Li, L. Chen, F. Song, B. Lin, Y. Cao, T. Liu, and Z. Sui. Making large language models better reasoners with alignment. arXiv preprint arXiv:2309.02144, 2023a.
P. Wang, L. Li, L. Chen, F. Song, B. Lin, Y. Cao, T. Liu, and Z. Sui. 通过对齐使大语言模型成为更好的推理器。arXiv preprint arXiv:2309.02144, 2023a.


P. Wang, L. Li, Z. Shao, R. Xu, D. Dai, Y. Li, D. Chen, Y. Wu, and Z. Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. CoRR, abs/2312.08935, 2023b.
P. Wang, L. Li, Z. Shao, R. Xu, D. Dai, Y. Li, D. Chen, Y. Wu, and Z. Sui. Math-Shepherd: 无需人类标注逐步验证并增强大语言模型。CoRR, abs/2312.08935, 2023b.


Z. Wang, R. Xia, and P. Liu. Generative AI for math: Part I - mathpile: A billion-token-scale pretraining corpus for math. CoRR, abs/2312.17120, 2023c. doi: 10.48550/ARXIV.2312.17120. URL https://doi.org/10.48550/arXiv.2312.17120.
Z. Wang, R. Xia, and P. Liu. 数学通用人工智能第一部分 - mathpile：一个十亿级 token 规模的数学预训练语料库. CoRR, abs/2312.17120, 2023c. doi: 10.48550/ARXIV.2312.17120. URL https://doi.org/10.48550/arXiv.2312.17120.


J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf 4f15af0f7b31abca4-Abstract-Conference.html
J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou. 思维链提示在大型语言模型中激发推理能力. In NeurIPS, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf 4f15af0f7b31abca4-Abstract-Conference.html


T. Wei, J. Luan, W. Liu, S. Dong, and B. Wang. Cmath: Can your language model pass chinese elementary school math test?, 2023.
T. Wei, J. Luan, W. Liu, S. Dong, and B. Wang. Cmath：你的语言模型能通过中国小学数学考试吗？, 2023.


M. Wenzel, L. C. Paulson, and T. Nipkow. The isabelle framework. In O. A. Mohamed, C. A. Muñoz, and S. Tahar, editors, Theorem Proving in Higher Order Logics, 21st International Conference, TPHOLs 2008, Montreal, Canada, August 18-21, 2008. Proceedings, volume 5170 of Lecture Notes in Computer Science, pages 33-38. Springer, 2008. doi: 10.1007/978-3-540-7 1067-7\\_7. URL https://doi.org/10.1007/978-3-540-71067-7_7
M. Wenzel, L. C. Paulson, and T. Nipkow. Isabelle 框架. In O. A. Mohamed, C. A. Muñoz, and S. Tahar, editors, Theorem Proving in Higher Order Logics, 21st International Conference, TPHOLs 2008, Montreal, Canada, August 18-21, 2008. Proceedings, volume 5170 of Lecture Notes in Computer Science, pages 33-38. Springer, 2008. doi: 10.1007/978-3-540-7 1067-7\\_7. URL https://doi.org/10.1007/978-3-540-71067-7_7


H. Xia, T. Ge, P. Wang, S.-Q. Chen, F. Wei, and Z. Sui. Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation. In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 3909- 3925, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/20 23.findings-emnlp.257. URL https://aclanthology.org/2023.findings-emnlp.257.
H. Xia, T. Ge, P. Wang, S.-Q. Chen, F. Wei, and Z. Sui. 投机解码：利用投机执行加速 seq2seq 生成. In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 3909- 3925, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/20 23.findings-emnlp.257. URL https://aclanthology.org/2023.findings-emnlp.257.


H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. arXiv preprint arXiv:2401.07851, 2024.
H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui. 解锁大型语言模型推理效率：投机解码综述. arXiv preprint arXiv:2401.07851, 2024.


S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.
S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan. 思维树：利用大型语言模型解决复杂问题. arXiv preprint arXiv:2305.10601, 2023.


L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, and W. Liu. Metamath: Bootstrap your own mathematical questions for large language models. CoRR, abs/2309.12284, 2023. doi: 10.48550/ARXIV.2309.12284. URL https://doi.org/10.485 50/arXiv.2309.12284
L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, and W. Liu. Metamath：为大型语言模型自举数学问题. CoRR, abs/2309.12284, 2023. doi: 10.48550/ARXIV.2309.12284. URL https://doi.org/10.485 50/arXiv.2309.12284


Z. Yuan, H. Yuan, C. Li, G. Dong, C. Tan, and C. Zhou. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023a.
Z. Yuan, H. Yuan, C. Li, G. Dong, C. Tan, and C. Zhou. 大型语言模型学习数学推理的规模化关系研究. arXiv preprint arXiv:2308.01825, 2023a.


Z. Yuan, H. Yuan, C. Tan, W. Wang, S. Huang, and F. Huang. Rrhf: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302, 2023b.
Z. Yuan, H. Yuan, C. Tan, W. Wang, S. Huang, and F. Huang. RRHF：通过响应排名轻松实现语言模型与人类反馈对齐. arXiv preprint arXiv:2304.05302, 2023b.


X. Yue, X. Qu, G. Zhang, Y. Fu, W. Huang, H. Sun, Y. Su, and W. Chen. Mammoth: Building math generalist models through hybrid instruction tuning. CoRR, abs/2309.05653, 2023. doi: 10.48550/ARXIV.2309.05653.URLhttps://doi.org/10.48550/arXiv.2309.05653.
X. Yue, X. Qu, G. Zhang, Y. Fu, W. Huang, H. Sun, Y. Su, and W. Chen. MAMMOTH：通过混合指令微调构建数学通用模型. CoRR, abs/2309.05653, 2023. doi: 10.48550/ARXIV.2309.05653.URLhttps://doi.org/10.48550/arXiv.2309.05653.


K. Zheng, J. M. Han, and S. Polu. Minif2f: a cross-system benchmark for formal olympiad-level mathematics. arXiv preprint arXiv:2109.00110, 2021.
K. Zheng, J. M. Han, and S. Polu. Minif2f: a cross-system benchmark for formal olympiad-level mathematics. arXiv preprint arXiv:2109.00110, 2021.


W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A. Saied, W. Chen, and N. Duan. AGIEval: A human-centric benchmark for evaluating foundation models. CoRR, abs/2304.06364, 2023. doi: 10.48550/arXiv.2304.06364. URLhttps://doi.org/10.48550/arXiv.2304.06364.
W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A. Saied, W. Chen, and N. Duan. AGIEval: A human-centric benchmark for evaluating foundation models. CoRR, abs/2304.06364, 2023. doi: 10.48550/arXiv.2304.06364. URLhttps://doi.org/10.48550/arXiv.2304.06364.


## A. Appendix
## A. 附录


### A.1. Analysis of Reinforcement Learning
### A.1. 强化学习分析


We provide the detailed derivation of the data source and gradient coefficient (algorithm and reward function) across various methods, including SFT, RFT, Online RFT, DPO, PPO, and GRPO.
我们提供了包括 SFT、RFT、在线 RFT、DPO、PPO 和 GRPO 在内的各种方法在数据源和梯度系数（算法和奖励函数）方面的详细推导。


#### A.1.1. Supervised Fine-tuning
#### A.1.1. 有监督微调


The objective of Supervised Fine-tuning is maximizing the following objective:
有监督微调的目标是最大化以下目标函数：


$$
{\mathcal{J}}_{SFT}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q,o \sim  {P}_{sft}\left( {Q,O}\right) }\right\rbrack  \left( {\frac{1}{\left| o\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| o\right| }\log {\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }\right) . \tag{6}
$$



The gradient of ${\mathcal{J}}_{SFT}\left( \theta \right)$ is:
${\mathcal{J}}_{SFT}\left( \theta \right)$ 的梯度为：


$$
{\nabla }_{\theta }{\mathcal{J}}_{SFT} = \mathbb{E}\left\lbrack  {q,o \sim  {P}_{sft}\left( {Q,O}\right) }\right\rbrack  \left( {\frac{1}{\left| o\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| o\right| }{\nabla }_{\theta }\log {\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }\right) . \tag{7}
$$



Data Source: The dataset employed for SFT. Reward Function: This can be regarded as human selection. Gradient Coefficient: always set to 1.
数据源：用于 SFT 的数据集。奖励函数：可视为人类选择。梯度系数：始终设置为 1。


#### A.1.2. Rejection Sampling Fine-tuning
#### A.1.2. 拒绝采样微调


Rejection Sampling Fine-tuning first samples multiple outputs from the supervised fine-tuned LLMs for each question, and then trains LLMs on the sampled outputs with the correct answer. Formally, the objective of RFT is to maximize the following objectives:
拒绝采样微调首先针对每个问题从有监督微调后的 LLM 中采样多个输出，然后使用其中答案正确的采样输出对 LLM 进行训练。形式上，RFT 的目标是最大化以下目标函数：


$$
{\mathcal{J}}_{RFT}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{sft}\left( {O \mid  q}\right) }\right\rbrack  \left( {\frac{1}{\left| o\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| o\right| }\mathbb{I}\left( o\right) \log {\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }\right) . \tag{8}
$$



The gradient of ${\mathcal{J}}_{RFT}\left( \theta \right)$ is:
${\mathcal{J}}_{RFT}\left( \theta \right)$ 的梯度为：


$$
{\nabla }_{\theta }{\mathcal{J}}_{RFT}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{sft}\left( {O \mid  q}\right) }\right\rbrack  \left( {\frac{1}{\left| o\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| o\right| }\mathbb{I}\left( o\right) {\nabla }_{\theta }\log {\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }\right) . \tag{9}
$$



Data Source: question in SFT dataset with outputs sampled from SFT model. Reward Function: Rule (whether the answer is correct or not). Gradient Coefficient:
数据源：SFT 数据集中的问题，输出采样自 SFT 模型。奖励函数：规则（答案是否正确）。梯度系数：


$$
G{C}_{RFT}\left( {q,o,t}\right)  = \mathbb{I}\left( o\right)  = \left\{  \begin{array}{rr} 1 & \text{ the answer of }o\text{ is correct } \\  0 & \text{ the answer of }o\text{ is incorrect } \end{array}\right. \tag{10}
$$



#### A.1.3. Online Rejection Sampling Fine-tuning
#### A.1.3. 在线拒绝采样微调


The only difference between RFT and Online RFT is that the outputs of Online RFT are sampled from the real-time policy model ${\pi }_{\theta }$ ,rather than from the SFT model ${\pi }_{{\theta }_{sft}}$ . Therefore,the gradient of online RFT is:
RFT 与在线 RFT 的唯一区别在于，在线 RFT 的输出采样自实时策略模型 ${\pi }_{\theta }$，而非 SFT 模型 ${\pi }_{{\theta }_{sft}}$。因此，在线 RFT 的梯度为：


$$
{\nabla }_{\theta }{\mathcal{J}}_{\text{ OnRFT }}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  {P}_{\text{ sft }}\left( Q\right) ,o \sim  {\pi }_{\theta }\left( {O \mid  q}\right) }\right\rbrack  \left( {\frac{1}{\left| o\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| o\right| }\mathbb{I}\left( o\right) {\nabla }_{\theta }\log {\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }\right) . \tag{11}
$$



#### A.1.4. Direct Preference Optimization (DPO)
#### A.1.4. 直接偏好优化 (DPO)


The objective of DPO is:
DPO 的目标函数为：


$$
{\mathcal{J}}_{DPO}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  {P}_{sft}\left( Q\right) ,{o}^{ + },{o}^{ - } \sim  {\pi }_{sft}\left( {O \mid  q}\right) }\right\rbrack  \log \sigma \left( {\beta \left| {\;\frac{1}{\left| {o}^{ + }\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| {o}^{ + }\right| }\log \frac{{\pi }_{\theta }\left( {{o}_{t}^{ + } \mid  q,{o}_{ < t}^{ + }}\right) }{{\pi }_{\mathrm{{ref}}}\left( {{o}_{t}^{ + } \mid  q,{o}_{ < t}^{ + }}\right) } - \beta \frac{1}{\left| {o}^{ - }\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| {o}^{ - }\right| }\log \frac{{\pi }_{\theta }\left( {{o}_{ < t}^{ - } \mid  q,{o}_{ < t}^{ - }}\right) }{{\pi }_{\mathrm{{ref}}}\left( {{o}_{ < t}^{ - } \mid  q,{o}_{ < t}^{ - }}\right) }}\right. }\right) \left( t\right) \tag{12}
$$



The gradient of ${\mathcal{J}}_{DPO}\left( \theta \right)$ is:
${\mathcal{J}}_{DPO}\left( \theta \right)$ 的梯度为：


(13)

$$
{\nabla }_{\theta }{\mathcal{J}}_{DPO}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  {P}_{sft}\left( Q\right) ,{o}^{ + },{o}^{ - } \sim  {\pi }_{sft}\left( {O \mid  q}\right) }\right\rbrack  \left( {\frac{1}{\left| {o}^{ + }\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| {o}^{ + }\right| }G{C}_{DPO}\left( {q,o,t}\right) {\nabla }_{\theta }\log {\pi }_{\theta }\left( {{o}_{t}^{ + } \mid  q,{o}_{ < t}^{ + }}\right) }\right.
$$

$$
- \frac{1}{\left| {o}^{ - }\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| {o}^{ - }\right| }G{C}_{DPO}\left( {q,o,t}\right) {\nabla }_{\theta }\log {\pi }_{\theta }\left( {{o}_{t}^{ - } \mid  q,{o}_{ < t}^{ - }}\right)
$$



Data Source: question in SFT dataset with outputs sampled from SFT model. Reward Function: human preference in the general domain (can be 'Rule' in mathematical tasks). Gradient Coefficient:
数据源：SFT 数据集中的问题，输出采样自 SFT 模型。奖励函数：通用领域的模型偏好（在数学任务中可以是“规则”）。梯度系数：


$$
G{C}_{DPO}\left( {q,o,t}\right)  = \sigma \left( {\beta \log \frac{{\pi }_{\theta }\left( {{o}_{t}^{ - } \mid  q,{o}_{ < t}^{ - }}\right) }{{\pi }_{\text{ ref }}\left( {{o}_{t}^{ - } \mid  q,{o}_{ < t}^{ - }}\right) } - \beta \log \frac{{\pi }_{\theta }\left( {{o}_{t}^{ + } \mid  q,{o}_{ < t}^{ + }}\right) }{{\pi }_{\text{ ref }}\left( {{o}_{t}^{ + } \mid  q,{o}_{ < t}^{ + }}\right) }}\right) \tag{14}
$$



#### A.1.5. Proximal Policy Optimization (PPO)
#### A.1.5. 近端策略优化 (PPO)


The objective of PPO is:
PPO 的目标函数为：


$$
{\mathcal{J}}_{PPO}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{{\theta }_{old}}\left( {O \mid  q}\right) }\right\rbrack  \frac{1}{\left| o\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| o\right| }\min \left\lbrack  {\frac{{\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }{{\pi }_{{\theta }_{old}}\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }{A}_{t},\operatorname{clip}\left( {\frac{{\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }{{\pi }_{{\theta }_{old}}\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) },1 - \varepsilon ,1 + \varepsilon }\right) {A}_{t}}\right\rbrack  . \tag{15}
$$



To simplify the analysis, it is assumed that the model only has a single update following each exploration stage,thereby ensuring that ${\pi }_{{\theta }_{\text{ old }}} = {\pi }_{\theta }$ . In this case,we can remove the min and clip operation:
为简化分析，假设模型在每个探索阶段后仅进行单次更新，从而确保 ${\pi }_{{\theta }_{\text{ old }}} = {\pi }_{\theta }$。在这种情况下，我们可以移除 min 和 clip 操作：


$$
{\mathcal{J}}_{PPO}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{{\theta }_{old}}\left( {O \mid  q}\right) }\right\rbrack  \frac{1}{\left| o\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| o\right| }\frac{{\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }{{\pi }_{{\theta }_{old}}\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) }{A}_{t}. \tag{16}
$$



The gradient of ${\mathcal{J}}_{PPO}\left( \theta \right)$ is:
${\mathcal{J}}_{PPO}\left( \theta \right)$ 的梯度为：


$$
{\nabla }_{\theta }{\mathcal{J}}_{PPO}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  {P}_{sft}\left( Q\right) ,o \sim  {\pi }_{{\theta }_{old}}\left( {O \mid  q}\right) }\right\rbrack  \frac{1}{\left| o\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| o\right| }{A}_{t}{\nabla }_{\theta }\log {\pi }_{\theta }\left( {{o}_{t} \mid  q,{o}_{ < t}}\right) \tag{17}
$$



Data Source: question in SFT dataset with outputs sampled from policy model. Reward Function: reward model. Gradient Coefficient:
数据源：SFT 数据集中的问题，输出采样自策略模型。奖励函数：奖励模型。梯度系数：


$$
G{C}_{PPO}\left( {q,o,t,{\pi }_{{\theta }_{rm}}}\right)  = {A}_{t}, \tag{18}
$$



where ${A}_{t}$ is the advantage,which is computed by applying Generalized Advantage Estimation (GAE) (Schulman et al. 2015),based on the rewards $\left\{  {r}_{ \geq  t}\right\}$ and a learned value function ${V}_{\psi }$ .
其中 ${A}_{t}$ 是优势值，基于奖励 $\left\{  {r}_{ \geq  t}\right\}$ 和学习到的价值函数 ${V}_{\psi }$，通过广义优势估计 (GAE) (Schulman et al. 2015) 计算得出。


#### A.1.6. Group Relative Policy Optimization (GRPO)
#### A.1.6. 群体相对策略优化 (GRPO)


The objective of GRPO is (assume ${\pi }_{{\theta }_{\text{ old }}} = {\pi }_{\theta }$ for simplified analysis):
GRPO 的目标函数为（为简化分析，假设 ${\pi }_{{\theta }_{\text{ old }}} = {\pi }_{\theta }$）：


$$
{\mathcal{J}}_{GRPO}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  {P}_{sft}\left( Q\right) ,{\left\{  {o}_{i}\right\}  }_{i = 1}^{G} \sim  {\pi }_{{\theta }_{old}}\left( {O \mid  q}\right) }\right\rbrack
$$



$$
\frac{1}{G}\mathop{\sum }\limits_{{i = 1}}^{G}\frac{1}{\left| {o}_{i}\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| {o}_{i}\right| }\left\lbrack  {\frac{{\pi }_{\theta }\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) }{{\pi }_{{\theta }_{\text{ old }}}\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) }{\widehat{A}}_{i,t} - \beta \left( {\frac{{\pi }_{\text{ ref }}\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) }{{\pi }_{\theta }\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) } - \log \frac{{\pi }_{\text{ ref }}\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) }{{\pi }_{\theta }\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) } - 1}\right) }\right\rbrack  . \tag{19}
$$



The gradient of ${\mathcal{J}}_{GRPO}\left( \theta \right)$ is:
${\mathcal{J}}_{GRPO}\left( \theta \right)$ 的梯度为：


$$
{\nabla }_{\theta }{\mathcal{J}}_{GRPO}\left( \theta \right)  = \mathbb{E}\left\lbrack  {q \sim  {P}_{sft}\left( Q\right) ,{\left\{  {o}_{i}\right\}  }_{i = 1}^{G} \sim  {\pi }_{{\theta }_{old}}\left( {O \mid  q}\right) }\right\rbrack
$$



$$
\frac{1}{G}\mathop{\sum }\limits_{{i = 1}}^{G}\frac{1}{\left| {o}_{i}\right| }\mathop{\sum }\limits_{{t = 1}}^{\left| {o}_{i}\right| }\left\lbrack  {{\widehat{A}}_{i,t} + \beta \left( {\frac{{\pi }_{ref}\left( {{o}_{i,t} \mid  {o}_{i, < t}}\right) }{{\pi }_{\theta }\left( {{o}_{i,t} \mid  {o}_{i, < t}}\right) } - 1}\right) }\right\rbrack  {\nabla }_{\theta }\log {\pi }_{\theta }\left( {{o}_{i,t} \mid  q,{o}_{i, < t}}\right) . \tag{20}
$$



Data Source: question in SFT dataset with outputs sampled from policy model. Reward Function: reward model. Gradient Coefficient:
数据源：SFT 数据集中的问题及策略模型采样的输出。奖励函数：奖励模型。梯度系数：


$$
G{C}_{GRPO}\left( {q,o,t,{\pi }_{{\theta }_{rm}}}\right)  = {\widehat{A}}_{i,t} + \beta \left( {\frac{{\pi }_{ref}\left( {{o}_{i,t} \mid  {o}_{i, < t}}\right) }{{\pi }_{\theta }\left( {{o}_{i,t} \mid  {o}_{i, < t}}\right) } - 1}\right) , \tag{21}
$$



where ${\widehat{A}}_{i,t}$ is computed based on the group reward scores.
其中 ${\widehat{A}}_{i,t}$ 是根据组奖励分数计算得出的。