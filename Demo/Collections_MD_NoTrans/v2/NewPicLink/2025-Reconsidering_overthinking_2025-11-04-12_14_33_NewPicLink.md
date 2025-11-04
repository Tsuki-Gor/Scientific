# Reconsidering Overthinking: Penalizing Internal and External Redundancy in CoT Reasoning

Jialiang Hong ${}^{1 * }$ ,Taihang Zhen ${}^{2 * }$ ,Kai Chen ${}^{2}$ ,Jiaheng Liu ${}^{2}$ ,Wenpeng Zhu ${}^{1 \dagger  }$ ,Jing Huo ${}^{2 \dagger  }$ ,Yang ${\mathrm{{Gao}}}^{{2}^{ \dagger  }}$ ,Depeng Wang ${}^{1}$ ,Haitao Wan ${}^{1}$ ,Xi Yang ${}^{1}$ ,Boyan Wang ${}^{2}$ ,Fanyu Meng ${}^{3}$

${}^{1}$ China Mobile (Suzhou) Software Technology Co.,Ltd.

${}^{2}$ State Key Laboratory for Novel Software Technology,Nanjing University

${}^{3}$ China Mobile Research Institute

## Abstract

Large Reasoning Models (LRMs) often produce excessively verbose reasoning traces, a phenomenon known as over-thinking, which hampers both efficiency and interpretability. Prior works primarily address this issue by reducing response length, without fully examining the underlying semantic structure of the reasoning process. In this paper, we revisit overthinking by decomposing it into two distinct forms: internal redundancy, which consists of low-contribution reasoning steps within the first correct solution (FCS), and external redundancy, which refers to unnecessary continuation after the FCS. To mitigate both forms, we propose a dual-penalty reinforcement learning framework. For internal redundancy, we adopt a sliding-window semantic analysis to penalize low-gain reasoning steps that contribute little toward reaching the correct answer. For external redundancy, we penalize its proportion beyond the FCS to encourage earlier termination. Our method significantly compresses reasoning traces with minimal accuracy loss, and generalizes effectively to out-of-domain tasks such as question answering and code generation. Crucially, we find that external redundancy can be safely removed without degrading performance, whereas internal redundancy must be reduced more cautiously to avoid impairing correctness. These findings suggest that our method not only improves reasoning efficiency but also enables implicit, semantic-aware control over Chain-of-Thought length, paving the way for more concise and interpretable LRMs.

Code - https://github.com/HenryZhen97/Reconsidering-Overthinking

## Introduction

Large reasoning models (LRMs), such as OpenAI's ol (Jaech et al. 2024), DeepSeek-R1 (Guo et al. 2025), and QwQ (Team 2025), demonstrate strong performance on complex reasoning tasks. A key factor behind their success is the generation of dense Chain-of-Thought (CoT) sequences, which decompose complex problems into stepwise reasoning that guides models toward correct answers. However, such lengthy reasoning chains often contain substantial redundancy, which can hinder the overall reasoning efficiency and reduce the readability of LRMs' outputs (Chen et al. 2024; Sui et al. 2025).

<!-- Media -->

<!-- figureText: redundancy Every morning Aya walks $9\mathrm{\;{km}}\ldots$ at $\mathrm{s}\mathrm{{km}}/\mathrm{h}$ the trip is $4\mathrm{\;h}$ (incl. t) $\ldots$ at $\mathrm{s}$ $+ 2\mathrm{\;{km}}/\mathrm{h}$ it is $2\mathrm{\;h}{24}\mathrm{\;{min}}$ (incl. t) ... if she walks at $\mathrm{s} + 1/2\mathrm{\;{km}}/\mathrm{h}$ ,how many minutes will the entire outing take (incl. t)? FCS redundancy Deepseek-R1 Ours Alright, let's break down the problem step by step. Problem Understanding: Aya walks 9 kilometers every morning. When she Information:1. Distance: $9\mathrm{\;{km}}$ 2. Times: At speed s: total time 4 hours (so time walking + 1 minutes) ... Step 1: Express the walking time in terms of speed. .. Step 2: Set up equations. ... Step 7 Calculate time when speed is $\mathrm{s} + {0.5}$ $= 3\mathrm{\;{km}}/\mathrm{h}$ ; Total time $=$ walking time +coffee time $= 3 + {24}/{60} = 3 + {0.4} =$ 3.4 hours. Convert to minutes: ${3.4} \times$ ${60} = \underline{\mathbf{{204}}}$ minutes. Final Answer: 204 $\mathbf{{GPT} - {o1}}$ Aya’s daily walk, $9\mathrm{\;{km}}$ distance Aya walks $9\mathrm{\;{km}}$ daily,with constant includes coffee time, constant : coffee break t, given two total times minutes, two speeds provided-s and at speeds s and s+2 km/h,goal is time s+2 km/h....I set up two equations at $\mathrm{s} + 1/2\mathrm{\;{km}}/\mathrm{h}$ . ... I set two equations first scenario: $9/\mathrm{s} + \mathrm{t} = 4\mathrm{\;h}$ ,second simplify to find s. ... Let's check scenario: $9/\left( {\mathrm{s} + 2}\right)  + \mathrm{t} = {2.4}\mathrm{h}$ . ...Let ${}^{\mathrm{t}}$ : algebra carefully: subtract equations carefully: subtracting to eliminate t, solve s : $2\mathrm{\;s} - {11.25} =$ equations to eliminate coffee time 0. obtaining $\mathrm{s} = {2.5}\mathrm{\;{km}}/\mathrm{h}$ . coffee $\mathrm{t} =$ finding $\mathrm{s} = {2.5}\mathrm{\;{km}}/\mathrm{h}$ ,coffee break $\mathrm{t} =$ 24 min. ... Compute new speed: s+% 24 min. ...New scenario speed: s + $= 3\mathrm{\;{km}}/\mathrm{h}$ ; walking time $= 9\mathrm{\;{km}} \div  3$ ${0.5} = 3\mathrm{\;{km}}/\mathrm{h}$ ,walking time: $9\mathrm{\;{km}} \div$ ; $\mathrm{{km}}/\mathrm{h} = 3\mathrm{\;h}$ . ... Add coffee time: $3\mathrm{\;h} +$ $\mathrm{{km}}/\mathrm{h} = 3\mathrm{\;h}$ . ...Total time $=$ walking : ${24}\mathrm{\;{min}} = \underline{204}$ minutes. Double check h + coffee 24 min = 204 minutes calculations again: previous scenario: Double check final calculation: 180 ${216} + {24} = {240}\mathrm{\;{min}},{120} + {24} = {144}\mathrm{\;{min}}$ min + 24 min coffee = 204 min new scenario ${180} + {24} = {204}\mathrm{\;{min}} - \mathrm{{al}}$ consistent—answer stable... good, final answer confirmed -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_42_54_20b98a.jpg"/>

Figure 1: Response examples in AIME24 from o1, R1, and our method. The underlined number "204" marks the first correct answer, which splits the reasoning into two parts: the preceding span is the First Correct Solution (FCS), which may contain internal redundancy; the continuation beyond it constitutes external redundancy.

<!-- Media -->

Recent efforts (Liu et al. 2025a; Sheng et al. 2025; Wang et al. 2025a,b) investigate overthinking in LRMs, aiming to uncover its root causes and develop targeted compression techniques for CoT outputs. Among them, a line of work grounded in reinforcement learning (RL) further explores how to suppress redundant reasoning by constraining token budgets (Luo et al. 2025a; Arora and Zanette 2025). However, these methods mainly target the symptom of redundancy by limiting the length of reasoning traces, without addressing its underlying semantic causes. In contrast, our work analyzes the semantic nature of redundancy and decomposes it into two distinct types for more precise detection and suppression.

---

<!-- Footnote -->

*Equal contribution

${}^{ \dagger  }$ Corresponding author: Wenpeng Zhu (zhuwen-peng@cmss.chinamobile.com), Jing Huo (huojing@nju.edu.cn) and Yang Gao (gaoy@nju.edu.cn)

<!-- Footnote -->

---

<!-- Meanless: のSNSのサービス・・・・・ここでのNSXを-->


The first correct solution (FCS) (Chen et al. 2024) refers to the earliest complete reasoning trace that leads to a correct answer. Theoretically, there exists a minimal subset within the FCS that is sufficient to solve the task. Based on this assumption, any content beyond this minimal subset, either superfluous steps within the FCS or additional content generated afterward, can be regarded as redundancy. As shown in Figure 1, we analyze outputs from two mainstream LRMs on AIME24. By identifying the FCS in each response, we distinguish between two types of redundancy: internal redundancy, which refers to unnecessary steps embedded within the FCS itself, and external redundancy, which arises after the correct answer has already been produced. These observations motivate a semantic-aware compression framework aimed at removing both types of redundancy rather than merely shortening sequences. The third subfigure in Figure 1 illustrates the effect of our approach, where both internal and external redundancy have been successfully removed.

Internal Redundancy Compression LRMs often repeat semantically similar content, such as reiterating premises or reassessing intermediate steps, which we define as internal redundancy. To quantify this, we apply a sliding-window semantic similarity measure to detect repetitive spans before the first correct answer. A penalty is then incorporated into the RL objective to promote stepwise utility and reduce informational overlap.

External Redundancy Compression Once the correct answer is produced, further continuation (e.g., re-deriving the answer or verifying previous steps) contributes little to problem-solving. We define this as external redundancy. we use a normalized redundancy proportion, the ratio of trailing length to total reasoning length, as a penalty in the RL reward. This encourages the model to stop reasoning promptly once the correct answer is reached.

To validate our approach, we conduct extensive reinforcement learning experiments across three mathematical benchmarks. Results show that both internal and external redundancy steadily decrease during training, while accuracy is largely preserved. Notably, our method achieves comparable semantic conciseness to human-written solutions without relying on hard length constraints, confirming the effectiveness of our dual-penalty framework in guiding efficient reasoning behavior. In conclusion, our contributions are summarized as follows:

- We provide the first systematic decomposition of CoT redundancy into internal and external components, offering a novel semantic-aware perspective on overthinking in LRMs.

- We develop an innovative sliding-window semantic similarity approach to detect and penalize low-informative reasoning steps within the FCS, enabling fine-grained internal redundancy mitigation.

- We introduce a normalized proportion-based metric to quantify external redundancy beyond the FCS, and apply a targeted penalty to discourage unnecessary continuation.

- Extensive experiments demonstrate that our method significantly compresses reasoning traces with minimal accuracy degradation. Further ablation studies reveal that the slight accuracy loss primarily stems from removing internal redundancy, while external redundancy can be reduced safely without harming performance.

## Related Work

CoT reasoning (Wei et al. 2022) has become a core technique for enhancing the step-by-step reasoning capabilities of LLMs. By decomposing complex questions into intermediate reasoning steps, CoT improves answer accuracy and transparency (Qiao et al. 2022). However, with increasing task complexity, generated CoT traces often become unnecessarily lengthy and redundant, reducing inference efficiency (Chen et al. 2024; Team et al. 2025).

Recent studies have attempted to compress CoT length through reinforcement learning with explicit length-based reward functions (Team et al. 2025; Arora and Zanette 2025; Shen et al. 2025; Qu et al. 2025; Yang, Lin, and Yu 2025; She et al. 2025; Hou et al. 2025). While effective to some extent, these approaches treat redundancy as a monolithic problem, overlook the root cause of redundancy and risk removing essential reasoning steps.

In contrast to prior works, our approach introduces a semantic-aware dual-penalty framework that structurally decomposes overthinking into internal redundancy and external redundancy. By applying targeted penalties to each, our method provides finer control over reasoning compression and interpretability. To our knowledge, this is the first work to explicitly isolate and address these two forms of redundancy in LLM reasoning traces.

## CoT Redundancy Detection

Redundant reasoning in LRMs can occur at different stages of the CoT process (Han et al. 2024; Liu et al. 2024; Ma et al. 2025). As shown in Figure 1, we observe that redundancy clusters either before or after the first correct answer, motivating a segmentation-based analysis.

We adopt a regular-expression-based strategy to locate the earliest sentence containing the final correct answer. This sentence serves as a boundary to divide the CoT sequence into two segments: a pre-answer segment comprising all reasoning steps up to the answer, and a post-answer segment starting after the sentence where the answer first appears. This segmentation enables separate analysis of repetition patterns within the FCS (internal redundancy) and beyond it (external redundancy).

## Internal Redundancy Degree (IRD)

In Figure 2a, we design a pipeline that first splits each CoT output into $N$ discrete sentences $\left\{  {{s}_{1},{s}_{2},\ldots ,{s}_{N}}\right\}$ . A sentence-level sliding window is then applied over this sequence to detect local redundancy. To adapt to CoT outputs of varying lengths, we dynamically set the window size and stride as fixed proportions of the total sentence count $N$ . Specifically,the window size is defined as $w = \lfloor {\alpha N}\rfloor$ ,and the stride as $t = \lfloor {\beta N}\rfloor$ ,where $\alpha  \in  \left( {0,1}\right) ,\beta  \in  \left( {0,\alpha }\right)$ . The dynamic window size enables the method to scale naturally with different reasoning lengths,while the constraint $\beta  < \alpha$ ensures overlapping windows, which smooths the semantic similarity signal and improves the robustness of redundancy estimation. Each window represents a contiguous reasoning segment, preserving local semantic coherence. For each segment, we compute sentence-level embeddings and calculate cosine similarity between adjacent windows:


$$
{\mathbf{e}}_{i} = {f}_{\text{embedding }}\left( {\mathcal{W}}_{i}\right)  \tag{1}
$$

$$
{\operatorname{sim}}_{i} = \max \left( {0,\cos \left( {{\mathbf{e}}_{i},{\mathbf{e}}_{i + 1}}\right) }\right)  \tag{2}
$$

$$
\operatorname{IRD} = \frac{1}{M}\mathop{\sum }\limits_{{i = 1}}^{M}{\operatorname{sim}}_{i} \tag{3}
$$

<!-- Media -->

<!-- figureText: prize if all four are chosen. The probability of winning the grand prize given winning a prize is $\mathbf{m}/\mathbf{n},\mathbf{m}$ and $\mathbf{n}$ coprime. Find $\mathbf{m} + \mathbf{n}$ Jen. ..picks4distinct numbers from $S = \{ 1,2,3,\ldots ,{10}\} \ldots 4$ numbers are randomly chosen from $S$ . She wins a prize if at least two of her numbers are among... the chosen numbers,and wins the gram External redundancy analysi ${LLM}$ GSM8K ${ERD} =$ (d) AIME24 (c) (e) Internal redundancy analysis Jen picks 4 numbers ... So the final answer GSM8K-Sample should be 116 Split by sentence is there something I'm overlooking here? For the grand prize,only one way matches,so $\ldots$ is $1/{210}$ . Am I missing a subtle point in the logic? ... IRD ${115} = 5 \times  {23}$ ,and $1\ldots$ coprime,so $\mathrm{m} + \mathrm{n} = 1 + {115} = {116}$ . Preservations Vindow Embedding (b) Similarity -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_42_54_d8016a.jpg"/>

Figure 2: Analysis of CoT redundancy. (a) Internal redundancy detection using sliding windows. (b) Semantic similarities of QwQ responses, QwQ-CoT+, and human-written solutions. (c) The Internal Redundancy Degree (IRD) of QwQ responses, QwQ-CoT+ and human references across GSM8K, MATH500 and AIME24. (d) The External Redundancy Degree (ERD) measurement using length proportion. (e) The ERD of QwQ responses across GSM8K, MATH500, and AIME24.

<!-- Media -->

where ${\mathcal{W}}_{i} = \left\{  {{s}_{i},{s}_{i + 1},\ldots ,{s}_{i + w - 1}}\right\}$ denotes the $i$ -th windowed segment, $M$ is the total number of windows. To focus on semantic redundancy, we clip negative similarity values to zero during computation. Since CoT reasoning inherently relies on maintaining semantic continuity, elevated local similarity often signals insufficient informational progression. We thus define the IRD as the average similarity across adjacent segments, where higher values indicate less efficient reasoning within the FCS.

To empirically validate our proposed IRD metric, we construct a high-quality CoT dataset, QwQ-CoT+, by refining the original QwQ-32B model responses with a well-designed prompt. The resulting traces retain correctness while substantially reducing redundancy, producing concise multi-step reasoning. We evaluate the correct outputs of QwQ, QwQ-CoT+, and human-written solutions using OpenAI's text-3-embedding-large (OpenAI 2024) model to compute the IRD (with $\alpha  = {0.1},\beta  = {0.05}$ ). As shown in Figure 2b and 2c, QwQ-CoT+ and human references consistently exhibit lower IRD values than QwQ responses across the GSM8K, MATH500, and AIME24 benchmarks, confirming the IRD metric's ability to detect redundancy in the FCS. Notably, the IRD scores of QwQ-CoT+ and human references remain significantly above zero, indicating that a moderate redundancy may be necessary for effective reasoning.

## External Redundancy Degree (ERD)

External redundancy, defined as content beyond the FCS in CoT, typically does not contribute to deriving the correct answer and can be considered uninformative. To quantify this, we propose the External Redundancy Degree (ERD), which measures the proportion of redundant content rather than absolute length, avoiding bias against longer CoT outputs. Specifically, ERD is calculated as the ratio of the length of the external segment after the FCS to the total length of the CoT reasoning trace:

$$
\mathrm{{ERD}} = \frac{{L}_{\text{external }}}{{L}_{\text{total }}} \tag{4}
$$

As illustrated in Figure 2d, a higher ERD indicates greater external redundancy in the CoT reasoning process. Since QwQ-CoT+ and human-written solutions exhibit no external redundancy, we report only QwQ's ERD performance in Figure 2e. This metric effectively quantifies unnecessary reasoning produced after reaching the first correct answer, providing a clear measure of model efficiency in CoT reasoning.


## Dual-Redundancy Penalty

To mitigate internal and external redundancy, we augment the reinforcement learning objective by applying two penalty terms to the accuracy-based reward. These penalties are computed from the internal and external redundancy degrees and encourage the model to generate concise and efficient reasoning without unnecessary repetition or continuation.

Internal Redundancy Penalty To reduce the redundancy in FCS, we apply a normalized internal redundancy penalty upon the accuracy reward. The total reward function is as follows:

$$
{\mathbb{E}}_{x \sim  D}\left\lbrack  {\mathbb{1}\left\{  {{y}_{\text{pred }} = {y}_{\mathrm{{GT}}}}\right\}   \cdot  \frac{{\sigma }_{k}\left( {1 - \mathrm{{IRD}}}\right)  - {\sigma }_{k}\left( 0\right) }{{\sigma }_{k}\left( 1\right)  - {\sigma }_{k}\left( 0\right) }}\right\rbrack  , \tag{5}
$$

where ${\sigma }_{k}\left( x\right)  = \frac{1}{1 + {e}^{-k\left( {x - c}\right) }}$ is a sharpened sigmoid function with slope $k$ and center $c$ . As shown in our earlier analysis of QwQ-CoT+ and human references, a moderate amount of internal redundancy is often essential for coherent reasoning. To preserve this desirable property, we calibrate the penalty function such that it only becomes active when the IRD exceeds a threshold. Specifically,we set $k = {20}$ and $c = {0.3}$ in this paper,so that the penalty is negligible when IRD $< {0.5}$ ,and increases rapidly beyond this point. This allows the model to tolerate reasonable redundancy while still discouraging excessive repetition.

External Redundancy Penalty Similarly, to discourage unnecessary continuation after the answer is found, we apply a normalized linear penalty based on the ERD:

$$
{\mathbb{E}}_{x \sim  D}\left\lbrack  {\mathbb{1}\left\{  {{y}_{\text{pred }} = {y}_{\mathrm{{GT}}}}\right\}   \cdot  \left( {1 - \mathrm{{ERD}}}\right) }\right\rbrack   \tag{6}
$$

## Experiment

In this section, we detail the experimental setup and comparative results to assess how well our method reduces over-thinking compared to existing RL-based length compression approaches. The overall training procedure is summarized in Algorithm 1.

## Training Setup

We adopt verl (Sheng et al. 2024), a scalable and high-throughput reinforcement learning library tailored for LLMs. Training is conducted using Group Relative Policy Optimization (GRPO) (Shao et al. 2024) across 64 NVIDIA A800 GPUs. We fine-tune DeepSeek-R1-Distilled-Qwen- 1.5B and DeepSeek-R1-Distill-Qwen-7B on a subset of DeepScaleR (Luo et al. 2025b), originally containing 40k math questions. This subset is curated by selecting ${10}\mathrm{k}$ samples whose answers are purely numeric and contain at least two digits to ensure accurate extraction of FCS from CoT traces. Notably, We choose DeepScaleR as our training corpus because it has been widely adopted in prior works on RL-based CoT compression and is also used, fully or partially, by several baselines we compare against. This ensures a fair and consistent training setup for evaluating the effectiveness of our proposed method. Our training is performed with temperature 0.6, top-p 1.0, 8 samples per input, and a batch size of 128 .

<!-- Media -->

Algorithm 1: Training with Dual-Redundancy Penalty

---

Require: Model $\mathcal{M}$ ,Dataset $\mathcal{D}$ ,Reward function $\mathcal{R}$ ,Opti-
		mizer $\mathcal{O}$ ,Window size $w$
		for each batch of prompts ${\left\{  {x}_{i}\right\}  }_{i = 1}^{N}$ from $\mathcal{D}$ do
			Sample $K$ responses ${\left\{  {\widehat{y}}_{i}^{\left( k\right) }\right\}  }_{k = 1}^{K}$ from $\mathcal{M}\left( {x}_{i}\right)$ for each
			${x}_{i}$
			for each response ${\widehat{y}}_{i}^{\left( k\right) }$ do
				if final answer is incorrect then
					Assign reward ${r}_{i}^{\left( k\right) } \leftarrow  0$
				else
					Locate first correct solution (FCS) in ${\widehat{y}}_{i}^{\left( k\right) }$
					Split into [FCS, post-FCS]
					Compute internal redundancy penalty: ${p}_{\text{int }}$
					Compute external redundancy penalty: ${p}_{\text{ext }}$
					Assign reward: ${r}_{i}^{\left( k\right) } \leftarrow  {r}_{i}^{\left( k\right) } \cdot  {p}_{\text{int }} \cdot  {p}_{\text{ext }}$
				end if
			end for
			Compute GRPO policy loss with $\left\{  {r}_{i}^{\left( k\right) }\right\}$
			Update model $\mathcal{M}$ via optimizer $\mathcal{O}$
		end for

---

<!-- Media -->

## Baselines

For a fair comparison, we benchmark our method against high-performing RL-based approaches that are representative of recent advances in length compression.

ThinkPrune (Hou et al. 2025) imposes a maximum generation length constraint during reinforcement learning, compelling the model to complete reasoning within a fixed token budget. This encourages concise reasoning behavior under pressure.

LC-R1 (Cheng et al. 2025) builds upon a length-penalty reward by introducing an additional compression reward, which is computed by using an auxiliary LLM to generate a compressed version of the original response. The reward is proportional to the reduction in length compared to the original.

Laser-DE (Liu et al. 2025b) avoids hard truncation by setting a context window significantly larger than the target length, and encourages brevity by assigning extra rewards to correct outputs whose lengths fall below the target length.

Training (Arora and Zanette 2025) leverages the multiple-sampling nature of reinforcement learning. It applies differentiated rewards based on the length of generated answers, favoring shorter completions that maintain correctness.

All baselines are evaluated using their publicly released models except for ThinkPrune's DeepSeek-R1- Distill-Qwen-7B model. As this model is not publicly available, we reproduced it following the implementation details described in their paper.

## Evaluation Setup

We conduct all evaluations under a unified experimental setup across three mathematical reasoning benchmarks of varying difficulty: GSM8K (Cobbe et al. 2021), MATH500 (Hendrycks et al. 2021), and AIME24. We use Pass@1 as the primary metric for reasoning accuracy. During inference, we allow a maximum response length of ${16}\mathrm{k}$ tokens. For GSM8K and MATH500, we sample 4 responses per instance with a temperature of 0.6 , while for AIME24, we use 64 samples due to the limited number of available problems. To ensure a fair comparison across different CoT compression methods, some of which may shorten or omit the conclusion, we exclude the final answer statement (conclusion) part from token length statistics. This allows us to more precisely measure the efficiency of the reasoning process itself.


<!-- Media -->

<!-- figureText: Response Length 4500 0.35 0.20 0.15 250 300 350 Training Step (a) 0.5 250 300 350 400 Training Step (b) 4000 3000 50 100 150 200 0.75 0.65 0.60 50 100 150 200 -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_42_54_22eb6c.jpg"/>

Figure 3: Training process. (a) Response length and reward trends. (b) IRD and ERD trends.

<!-- Media -->

## Token Efficiency Metric

In addition, to quantitatively evaluate the trade-off between reasoning accuracy and conciseness, we propose the Token Efficiency (TE) metric, defined as:

$$
\mathrm{{TE}} = \frac{A}{\sqrt{L}} \tag{7}
$$

This metric is inspired by the per-token accuracy formulation $A/L$ ,which captures how much each token contributes to the overall correctness. To reflect a stronger preference for accuracy over brevity during evaluation, we apply a square root to the token count $L$ ,dampening the penalty for longer but correct traces. A higher TE indicates a better balance between precision and efficiency.

## Main Results

We first present the training process of our dual-penalty method. Figure 3 shows that the reward steadily increases throughout training, while both IRD and ERD consistently decrease, demonstrating the effectiveness of the proposed penalty mechanism. Specifically, ERD converges to approximately 0.1 , and IRD stabilizes around 0.6 , comparable to the levels observed in QwQ-CoT+ and human-written solutions. This reduction is also reflected in the declining average response length, suggesting that our method encourages more efficient reasoning without relying on explicit length constraints. These findings provide a key insight: by targeting the semantic characteristics of redundant reasoning content, our approach enables LRMs to suppress overthinking in a principled and interpretable manner.

As shown in Table 1, our method achieves the best overall performance in terms of the token efficiency on both DeepSeek-R1-Distill-Qwen-1.5B and 7B models. Especially, on the more challenging MATH500 and AIME24 benchmarks, our approach significantly outperforms all baselines. In addition, we find that some existing length-compression methods still exhibit considerable internal and external redundancy in the reasoning traces, indicating that our method can be applied on top of existing compression techniques to further refine the CoT process. Among the baselines, those with higher TE scores also demonstrate marked reductions in both types of redundancy, suggesting a convergent trend: models effectively learn to compress CoT reasoning by minimizing internal and external redundancy. This observation indirectly validates that our method successfully captures the fundamental nature of redundancy in LRMs.

## Cross-Domain Generalization

<!-- Media -->

<!-- figureText: 6807 12237 31,49 13 -Owen-1.5B -Qwen-7E (b) LiveCodeBench 35.89 34 -Qwen-1.5B -Qwen-7B (a) GPQA Diamond -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_42_54_971f32.jpg"/>

Figure 4: Performance on GPQA Diamond and Live-CodeBench. Our method generalizes well to out-of-domain (OOD) reasoning tasks.

<!-- Media -->

Some findings (Meng et al. 2025; Chu et al. 2025; Xie et al. 2025) suggest that reinforcement learning can endow large models with strong generalization capabilities on out-of-domain (OOD) data. Motivated by this, we investigate whether the redundancy compression patterns learned through RL on mathematical reasoning tasks can generalize to non-mathematical domains. Specifically, we evaluate our trained models on two OOD benchmarks: GPQA Diamond (Rein et al. 2024), which emphasizes graduate-level factual reasoning, and LiveCodeBench (Jain et al. 2024), which targets multi-hop reasoning in program understanding. Models are evaluated on LiveCodeBench with test data collected between May 2023 and January 2025.

As shown in Figure 4, our method remains effective in compressing CoT traces across both datasets. This indicates that the model has internalized a domain-agnostic paradigm for generating concise and efficient reasoning traces, beyond merely overfitting to the training distribution. The results underscore the transferability of our RL-based compression framework and suggest its broader applicability to various reasoning-intensive tasks.

## Ablation Studies

Composition To rigorously evaluate the individual and combined contributions of internal and external redundancy penalties to CoT length compression, we perform an ablation study under controlled conditions. As a baseline, we apply a fixed truncation of $6\mathrm{k}$ tokens to each response during training. Building upon this, we separately introduce the internal redundancy penalty and the external redundancy penalty to assess their isolated impact on the reduction of response length over training steps. As shown in Figure 5a, both penalties independently lead to a substantial decrease in response length, reaching a minimal length around 3100 tokens. When both penalties are applied jointly, the response length is further compressed to approximately 2600 tokens, indicating a complementary and synergistic effect. Moreover,in Figure $5\mathrm{\;b}$ and $5\mathrm{c}$ ,we observe that the application of one penalty does not significantly influence the redundancy degree targeted by the other, suggesting that internal and external redundancy reflect orthogonal aspects of reasoning traces.


<!-- Media -->

<table><tr><td rowspan="2">Model</td><td colspan="5">GSM8K</td><td colspan="5">MATH500</td><td colspan="5">AIME24</td></tr><tr><td>Accuracy</td><td>Tokens</td><td>IRD</td><td>ERD</td><td>TE</td><td>Accuracy</td><td>Tokens</td><td>IRD</td><td>ERD</td><td>TE</td><td>Accuracy</td><td>Tokens</td><td>IRD</td><td>ERD</td><td>TE</td></tr><tr><td colspan="16">DeepSeek-R1-Distill-Qwen-1.5B</td></tr><tr><td>Original Model</td><td>84.1</td><td>1555</td><td>73.7</td><td>43.0</td><td>2.13</td><td>82.2</td><td>3549</td><td>77.5</td><td>55.2</td><td>1.38</td><td>28.5</td><td>8681</td><td>71.4</td><td>28.6</td><td>0.31</td></tr><tr><td>ThinkPrune-4k</td><td>86.1</td><td>910</td><td>77.9</td><td>40.1</td><td>2.85</td><td>83.7</td><td>2101</td><td>73.2</td><td>39.8</td><td>1.83</td><td>28.6</td><td>6431</td><td>75.2</td><td>21.0</td><td>0.36</td></tr><tr><td>LC-R1</td><td>82.5</td><td>507</td><td>67.2</td><td>19.3</td><td>3.66</td><td>79.6</td><td>1673</td><td>75.8</td><td>22.5</td><td>1.95</td><td>24.2</td><td>5075</td><td>79.6</td><td>20.4</td><td>0.34</td></tr><tr><td>Laser-DE</td><td>86.4</td><td>971</td><td>74.3</td><td>37.5</td><td>2.77</td><td>83.6</td><td>2282</td><td>78.0</td><td>36.3</td><td>1.75</td><td>32.7</td><td>7268</td><td>73.5</td><td>22.2</td><td>0.38</td></tr><tr><td>Training</td><td>81.0</td><td>292</td><td>61.6</td><td>7.8</td><td>4.74</td><td>82.8</td><td>1543</td><td>65.5</td><td>14.5</td><td>2.11</td><td>28.5</td><td>7049</td><td>73.2</td><td>17.4</td><td>0.34</td></tr><tr><td>$\mathbf{{Ours}}$</td><td>84.9</td><td>513</td><td>49.6</td><td>5.7</td><td>3.75</td><td>83.8</td><td>1505</td><td>51.0</td><td>7.9</td><td>2.16</td><td>34.0</td><td>6077</td><td>72.5</td><td>10.9</td><td>0.44</td></tr><tr><td colspan="16">DeepSeek-R1-Distill-Owen-7B</td></tr><tr><td>Original Model</td><td>91.1</td><td>844</td><td>70.0</td><td>36.0</td><td>3.14</td><td>91.2</td><td>2836</td><td>78.1</td><td>51.6</td><td>1.71</td><td>52.3</td><td>7241</td><td>77.8</td><td>31.1</td><td>0.61</td></tr><tr><td>ThinkPrune-4k</td><td>92.8</td><td>716</td><td>70.5</td><td>36.0</td><td>3.47</td><td>89.7</td><td>1683</td><td>77.9</td><td>36.1</td><td>2.19</td><td>50.4</td><td>5723</td><td>79.2</td><td>14.6</td><td>0.67</td></tr><tr><td>LC-R1</td><td>87.5</td><td>152</td><td>61.8</td><td>4.9</td><td>7.10</td><td>87.5</td><td>1201</td><td>65.8</td><td>7.0</td><td>2.52</td><td>52.7</td><td>6087</td><td>79.1</td><td>10.2</td><td>0.68</td></tr><tr><td>Laser-DE</td><td>93.3</td><td>637</td><td>68.2</td><td>31.1</td><td>3.70</td><td>92.1</td><td>1402</td><td>77.0</td><td>30.1</td><td>2.46</td><td>52.7</td><td>5061</td><td>80.5</td><td>11.8</td><td>0.74</td></tr><tr><td>Training</td><td>91.2</td><td>387</td><td>65.1</td><td>14.6</td><td>4.64</td><td>91.0</td><td>2090</td><td>76.3</td><td>38.3</td><td>1.99</td><td>50.8</td><td>6669</td><td>78.8</td><td>23.1</td><td>0.62</td></tr><tr><td>$\mathbf{{Ours}}$</td><td>90.9</td><td>318</td><td>51.8</td><td>6.5</td><td>5.10</td><td>89.8</td><td>1200</td><td>58.7</td><td>6.1</td><td>2.59</td><td>53.2</td><td>5025</td><td>77.4</td><td>3.7</td><td>0.75</td></tr></table>

Table 1: Comparison with CoT compression baselines. Accuracy denotes Pass@1 accuracy; Tokens indicates average CoT length. Our method achieves the best balance of accuracy and efficiency. We time IRD, ERD and TE with a scaling factor 100 for readability.

<!-- figureText: truncate-6k truncate-6k+internal truncate-6k+external truncate-6k+internal+external 0.4 ERD 0.2 0.1 200 300 400 100 200 300 400 Step Step (b) (c) 0.80 4500 0.75 Response Length 4000 0.70 IRD 3500 0.65 3000 0.60 0.55 0 100 200 300 400 100 Step (a) -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_42_54_1ce859.jpg"/>

Figure 5: Impact of internal and external redundancy penalties on CoT compression. These two penalties operate independently with minimal interference, yet their combined use enhances compression efficiency beyond individual applications.

<!-- Media -->

Accuracy To understand the source of accuracy degradation during CoT compression, we conduct a controlled ablation study that separately examines the effects of internal and external redundancy reduction. Unlike our main experiment where the base model (DeepSeek-R1-Distill-Qwen- 1.5B) still retains room for accuracy improvement during reinforcement learning, we deliberately choose a saturated model, DeepScaleR-1.5B-Preview, whose performance has plateaued. This allows us to eliminate potential reward-induced accuracy gains, thereby studying the true impact of redundancy compression.

We first apply external redundancy penalty alone during RL training,using a ${16}\mathrm{\;k}$ maximum response length. Once the ERD converges, we proceed to a second training stage where only the internal redundancy penalty is applied. In this phase,we limit the response length to $8\mathrm{k}$ tokens to avoid the model mistakenly adapting to longer outputs. All other training and evaluating configurations are kept consistent with the main experiment. We report the accuracy associated with various IRD and ERD checkpoints in Table 2.

The results show that when ERD converges to 0.09 , the model maintains nearly identical accuracy to its initial state across all three benchmarks (GSM8K, MATH500, and AIME24). In contrast, as the IRD is progressively reduced, accuracy drops substantially, especially on the more Table 2: Accuracy Changes under IRD and ERD Reduction. When ERD converges, the accuracy remains largely unaffected. In contrast, reducing IRD leads to notable accuracy drop, especially on the more challenging AIME24 benchmark. complex AIME24 dataset. These findings suggest that accuracy degradation during CoT compression is primarily attributable to the removal of internal redundancy.


<!-- Media -->

<table><tr><td rowspan="2">Model</td><td colspan="3">GSM8K</td><td colspan="3">MATH500</td><td colspan="3">AIME24</td></tr><tr><td>Accuracy</td><td>Tokens</td><td>Accuracy Drop</td><td>Accuracy</td><td>Tokens</td><td>Accuracy Drop</td><td>Accuracy</td><td>Tokens</td><td>Accuracy Drop</td></tr><tr><td>DeepScaleR-1.5B-Preview</td><td>87.8</td><td>1437</td><td>-</td><td>87.7</td><td>2593</td><td>-</td><td>41.1</td><td>7561</td><td>-</td></tr><tr><td>+ External penalty (ERD=0.09)</td><td>87.2</td><td>959</td><td>↓ 0.68%</td><td>87.6</td><td>1970</td><td>↓0.11%</td><td>41.0</td><td>6681</td><td>↓0.24%</td></tr><tr><td>+ Internal penalty (IRD=0.76)</td><td>87.2</td><td>762</td><td>↓ 0.68%</td><td>86.7</td><td>1758</td><td>↓1.14%</td><td>38.4</td><td>6555</td><td>↓ 6.57%</td></tr><tr><td>+ Internal penalty (IRD=0.68)</td><td>86.5</td><td>445</td><td>↓ 1.48%</td><td>84.0</td><td>1454</td><td>↓ 4.22%</td><td>38.2</td><td>5504</td><td>↓ 7.06%</td></tr><tr><td>+ Internal penalty (IRD=0.60)</td><td>85.4</td><td>345</td><td>↓ 2.73%</td><td>83.5</td><td>1247</td><td>↓ 4.79%</td><td>34.7</td><td>4810</td><td>↓ 15.57%</td></tr></table>

<!-- figureText: GSM8K MATH500 AIME24 0.43 0.42 0.40 0.39 0.20 0.10 0.20 0.15 0.12 0.08 ERD ERD (b) (c) MATH500 AIME24 0.41 0.39 0.36 0.33 IRD (f) 0.89 0.90 Accuracy 0.88 Accuracy 0.88 0.87 0.86 0.86 0.40 0.30 0.20 0.10 0.40 0.30 ERD (a) GSM8K 0.89 0.88 Accuracy Accuracy 0.86 0.83 0.84 0.80 IRD (d) -->

<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/11/2025_11_04__12_42_54_87597c.jpg"/>

Figure 6: Impact of IRD and ERD reduction on accuracy. Reducing IRD consistently lowers accuracy, whereas penalizing external redundancy does not harm performance.

<!-- Media -->

To further investigate this relationship, we examine how accuracy varies as a function of ERD reduction during the first stage of training. As shown in Figure 6a, 6b, and 6c, while a slight accuracy drop of approximately 2 percentage points is observed at the beginning of the training, the accuracy eventually returns to and remains at the initial level across GSM8K, MATH500, and AIME24. Given the inherent variance in evaluation accuracy, these results indicate that reducing external redundancy does not degrade the model's reasoning capability. This suggests that the portion of the reasoning trace following the FCS contributes little to final prediction correctness, highlighting the safety and effectiveness of external redundancy removal.

Similarly, we assess how accuracy responds to decreasing IRD in the second stage. As results shown in Figure 6d, 6e, and 6f, accuracy drops significantly as IRD decreases. This demonstrates that overly compressing the internal reasoning trace within the FCS directly harms model performance on tasks requiring fine-grained, multi-step reasoning. We hypothesize that during internal redundancy compression, the model is encouraged to eliminate intermediate reasoning steps, which increases the semantic gap between adjacent segments. This disrupts local coherence and leads to discontinuous reasoning trajectories or even CoT leaps that exceed the model's inference capability, ultimately resulting in reduced accuracy. This aligns with recent findings in (Xu et al. 2025). These observations highlight the critical importance of preserving a minimal level of internal reasoning structure and point to the need for adaptive compression strategies that balance brevity and coherence.

## Conclusion

In this paper, we introduce a novel view of overthinking in LRMs by decomposing it into internal and external redundancy, and propose a dual-penalty reinforcement learning framework to reduce both. Experiments show that this approach significantly reduces reasoning length with minimal accuracy loss, and that external redundancy can be safely removed without harming performance. We believe these insights offer a promising direction for developing more efficient and interpretable reasoning in LRMs. References


Arora, D.; and Zanette, A. 2025. Training Language Models to Reason Efficiently. arXiv preprint arXiv:2502.04463.

Chen, X.; Xu, J.; Liang, T.; He, Z.; Pang, J.; Yu, D.; Song, L.; Liu, Q.; Zhou, M.; Zhang, Z.; et al. 2024. Do not think that much for $2 + 3 = ?$ on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187.

Cheng, Z.; Chen, D.; Fu, M.; and Zhou, T. 2025. Optimizing Length Compression in Large Reasoning Models. arXiv preprint arXiv:2506.14755.

Chu, T.; Zhai, Y.; Yang, J.; Tong, S.; Xie, S.; Schuurmans, D.; Le, Q. V.; Levine, S.; and Ma, Y. 2025. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161.

Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; Hesse, C.; and Schulman, J. 2021. Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Han, T.; Wang, Z.; Fang, C.; Zhao, S.; Ma, S.; and Chen, Z. 2024. Token-budget-aware llm reasoning. arXiv preprint arXiv:2412.18547.

Hendrycks, D.; Burns, C.; Kadavath, S.; Arora, A.; Basart, S.; Tang, E.; Song, D.; and Steinhardt, J. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Hou, B.; Zhang, Y.; Ji, J.; Liu, Y.; Qian, K.; Andreas, J.; and Chang, S. 2025. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv:2504.01296.

Jaech, A.; Kalai, A.; Lerer, A.; Richardson, A.; El-Kishky, A.; Low, A.; Helyar, A.; Madry, A.; Beutel, A.; Carney, A.; et al. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720.

Jain, N.; Han, K.; Gu, A.; Li, W.-D.; Yan, F.; Zhang, T.; Wang, S.; Solar-Lezama, A.; Sen, K.; and Stoica, I. 2024. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974.

Liu, K.; Shen, C.; Zhang, Z.; Liu, J.; Yuan, X.; and ye, J. 2025a. Efficient Reasoning Through Suppression of Self-Affirmation Reflections in Large Reasoning Models. arXiv:2506.12353.

Liu, T.; Guo, Q.; Hu, X.; Jiayang, C.; Zhang, Y.; Qiu, X.; and Zhang, Z. 2024. Can language models learn to skip steps? arXiv preprint arXiv:2411.01855.

Liu, W.; Zhou, R.; Deng, Y.; Huang, Y.; Liu, J.; Deng, Y.; Zhang, Y.; and He, J. 2025b. Learn to Reason Efficiently with Adaptive Length-based Reward Shaping. arXiv preprint arXiv:2505.15612.

Luo, H.; Shen, L.; He, H.; Wang, Y.; Liu, S.; Li, W.; Tan, N.; Cao, X.; and Tao, D. 2025a. O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning. arXiv preprint arXiv:2501.12570.

Luo, M.; Tan, S.; Wong, J.; Shi, X.; Tang, W. Y.; Roongta, M.; Cai, C.; Luo, J.; Zhang, T.; Li, L. E.; et al. 2025b. Deep-scaler: Surpassing o1-preview with a ${1.5}\mathrm{\;b}$ model by scaling rl. Notion Blog.

Ma, X.; Wan, G.; Yu, R.; Fang, G.; and Wang, X. 2025. CoT-Valve: Length-Compressible Chain-of-Thought Tuning. arXiv preprint arXiv:2502.09601.

Meng, F.; Du, L.; Liu, Z.; Zhou, Z.; Lu, Q.; Fu, D.; Han, T.; Shi, B.; Wang, W.; He, J.; et al. 2025. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365.

OpenAI. 2024. text-embedding-3-large (Embedding Model). https://platform.openai.com/docs/models/text-embedding-3-large.

Qiao, S.; Ou, Y.; Zhang, N.; Chen, X.; Yao, Y.; Deng, S.; Tan, C.; Huang, F.; and Chen, H. 2022. Reasoning with language model prompting: A survey. arXiv preprint arXiv:2212.09597.

Qu, Y.; Yang, M. Y.; Setlur, A.; Tunstall, L.; Beeching, E. E.; Salakhutdinov, R.; and Kumar, A. 2025. Optimizing test-time compute via meta reinforcement fine-tuning. arXiv preprint arXiv:2503.07572.

Rein, D.; Hou, B. L.; Stickland, A. C.; Petty, J.; Pang, R. Y.; Dirani, J.; Michael, J.; and Bowman, S. R. 2024. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In First Conference on Language Modeling.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

She, J.; Li, Z.; Huang, Z.; Li, Q.; Xu, P.; Li, H.; and Ho, Q. 2025. Hawkeye: Efficient reasoning with model collaboration. arXiv preprint arXiv:2504.00424.

Shen, Y.; Zhang, J.; Huang, J.; Shi, S.; Zhang, W.; Yan, J.; Wang, N.; Wang, K.; Liu, Z.; and Lian, S. 2025. Dast: Difficulty-adaptive slow-thinking for large reasoning models. arXiv preprint arXiv:2503.04472.

Sheng, G.; Zhang, C.; Ye, Z.; Wu, X.; Zhang, W.; Zhang, R.; Peng, Y.; Lin, H.; and Wu, C. 2024. HybridFlow: A Flexible and Efficient RLHF Framework. arXiv preprint arXiv: 2409.19256.

Sheng, L.; Zhang, A.; Wu, Z.; Zhao, W.; Shen, C.; Zhang, Y.; Wang, X.; and Chua, T.-S. 2025. On Reasoning Strength Planning in Large Reasoning Models. arXiv preprint arXiv:2506.08390.

Sui, Y.; Chuang, Y.-N.; Wang, G.; Zhang, J.; Zhang, T.; Yuan, J.; Liu, H.; Wen, A.; Zhong, S.; Chen, H.; et al. 2025. Stop overthinking: A survey on efficient reasoning for large language models. arXiv preprint arXiv:2503.16419.

Team, K.; Du, A.; Gao, B.; Xing, B.; Jiang, C.; Chen, C.; Li, C.; Xiao, C.; Du, C.; Liao, C.; et al. 2025. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599.


Team, Q. 2025. QwQ-32B: Embracing the Power of Reinforcement Learning.

Wang, C.; Feng, Y.; Chen, D.; Chu, Z.; Krishna, R.; and Zhou, T. 2025a. Wait, We Don't Need to" Wait"! Removing Thinking Tokens Improves Reasoning Efficiency. arXiv preprint arXiv:2506.08343.

Wang, Y.; Liu, Q.; Xu, J.; Liang, T.; Chen, X.; He, Z.; Song, L.; Yu, D.; Li, J.; Zhang, Z.; et al. 2025b. Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs. arXiv preprint arXiv:2501.18585.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35: 24824-24837.

Xie, Y.; Ma, Y.; Lan, S.; Yuille, A.; Xiao, J.; and Wei, C. 2025. Play to Generalize: Learning to Reason Through Game Play. arXiv preprint arXiv:2506.08011.

Xu, H.; Yan, Y.; Shen, Y.; Zhang, W.; Hou, G.; Jiang, S.; Song, K.; Lu, W.; Xiao, J.; and Zhuang, Y. 2025. Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning. arXiv preprint arXiv:2505.14684.

Yang, J.; Lin, K.; and Yu, X. 2025. Think when you need: Self-adaptive chain-of-thought learning. arXiv preprint arXiv:2504.03234.