

<!-- Meanless: したので、今回はそのようなので、このことに、このようなので、今回はなくなり、それです。それです。それですね。-->

# A Self-Evolving Framework for Multi-Agent Medical Consultation Based on Large Language Models

Kai Chen ${}^{1}$ , Ji Qi ${}^{2}$ , Jing Huo ${}^{1 * }$ , Pinzhuo Tian ${}^{4}$ , Fanyu Meng ${}^{3}$ , Xi Yang ${}^{2}$ , Yang Gao ${}^{1}$ ${}^{1}$ State Key Laboratory for Novel Software Technology,Nanjing University,Nanjing,China ${}^{2}$ China Mobile (Suzhou) Software Technology Co.,Ltd. Suzhou,China ${}^{3}$ China Mobile Research Institute,Beijing,China

${}^{4}$ School of Computer Engineering and Science,Shanghai University,Shanghai,China

Abstract-We propose a multi-agent approach (SeM-Agents) based on large language models for medical consultations. This framework incorporates various doctor roles and auxiliary roles, with agents communicating through natural language. Using a residual structure, the system conducts multi-round medical consultations based on the patient's treatment background and symptoms. In the final summary and output stage of the consultation, it utilizes two experience databases-the Correct Consultation Experience Database and the Chain of Thought (CoT) Experience Database-which evolve with accumulated experience during consultations. This evolution drives the framework's self-improvement, significantly enhancing the rationality and accuracy of the consultations. To ensure that the conclusions are safe, reliable, and aligned with human values, the final decisions undergo a safety review before being provided to the patient. This framework achieved accuracy rates of ${89.2}\%$ and ${83.1}\%$ on the MedQA and PubMedQA datasets,respectively.

Index Terms-Large Language Model, Multi-Agent, Self-Evoling, Medical Consultation

## I. INTRODUCTION

Large Language Models (LLMs), with their extensive parameters and broad training across multiple domain knowledge bases, exhibit remarkable task generalization capabilities [1]- [5]. Their prowess in logical reasoning and complex problem-solving positions them as potential assets in medical diagnostics [6]-[8]. However, the acquisition of real medical consultation data is challenging due to privacy constraints and irregular data preservation, and even after medical knowledge fine-tuning, LLMs still suffer from hallucination issues [9], [10]. In the medical domain, any errors induced by hallucinations are unacceptable and can lead to severe medical mishaps. Autonomous agents driven by LLMs open new avenues for medical consultations. LLM-based multi-agent technologies not only enhance medical reasoning capabilities [11] but also more effectively elicit embedded medical knowledge, which cannot be solely guided by Chain of Thought (CoT) reasoning. Moreover, the complementary actions among agents in multi-round interactions significantly reduce the risk of hallucinations [12].

On one hand, well-designed organizational structures in Multi-Agent Systems significantly reduce system errors and enhance interaction efficiency [13]-[16]. ChatDev [13] decomposes tasks into sub-tasks, each managed by an instructor agent and an assistant agent, addressing software development and scientific discussion issues through multi-round inquiry-based collaboration, thus mitigating hallucinations. MAC-NET [16] organizes agents using a directed acyclic graph and simplifies their interactive reasoning through topological sorting to extract solutions from dialogues. MetaGPT [17] assigns agents roles within a software company and encodes Standard Operating Procedures (SOPs), demonstrating how to integrate agents' expertise to efficiently complete tasks. However, this method is primarily designed for software development, and its sequential execution process appears inefficient and unintuitive in medical consultation. Medagents [18] adopts a method where each LLM-Agent assumes a different doctor role, ultimately deciding the final solution through consensus voting. While this method is intuitive and clear, a simple voting mechanism without a robust organizational strategy could lead to collective hallucinations [19]. Moreover, these methods utilize static structures, unable to evolve, relying solely on the Zero-Shot capabilities of medical large models, which inherently limits their potential.

On the other hand, inspired by the ways human intelligence is acquired, the phenomenon of enhancing LLM-Agents' problem-solving capabilities by granting them memory experiences for reflection and application has been proven feasible [20]. ExpeL [21] accumulates experiences from past successes and applies this experiential knowledge during reasoning. ECL [22] concentrates on gathering experience-driven shortcuts derived from previous actions, which equips agents to more adeptly manage novel tasks. IER [23] allows LLM agents to iteratively refine experiences during task execution. Selfevolve [24] utilizes LLMs both as knowledge providers and as self-reflective programmers; through such reflective processes, agents can self-evolve. Agent Hospital [25] uses a Medical Record Library and an Experience Base to continuously accumulate diagnostic data, enhancing prompts for medical agents, and facilitating the evolution of medical agents.However, these efforts do not abstract, summarize, and reflect on erroneous cases, thus making it difficult to leverage valuable experiences from mistakes.

---

<!-- Footnote -->

This work was supported in part by Nanjing University - China Mobile Communications Group Co., Ltd. Joint Institute, in part by the Science and Technology Innovation 2030 New Generation Artificial Intelligence Major Project under Grant 2021ZD0113303; in part by the National Natural Science Foundation of China under Grant 62192783, Grant 62276128; in part by the Natural Science Foundation of Jiangsu Province under Grant BK20243051; in part by the Collaborative Innovation Center of Novel Software Technology and Industrialization.

<!-- Footnote -->

---

<!-- Meanless: Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on October 27,2025 at 02:45:21 UTC from IEEE Xplore. Restrictions apply.-->


<!-- Media -->

<!-- figureText: Arrange Doctors (A) ✘ Correct Answer Knowledge Base Shared Vector Database Summarize (C) Output Reach a consensus Chain of Thought Knowledge Base Roles Select Doctors (B) A ✘ If $=$ No consensus XN B/D D D During Medical Consultation -->

<img src="https://cdn.noedgeai.com/bo_d3vdvik601uc738lu9u0_1.jpg?x=339&y=139&w=1118&h=612&r=0"/>

Fig. 1. Overview of the Medical Consultation Framework: (A) Arranging specialist doctors based on the specific situation of the patient; (B) Multi-round consultations of expert Agents; (C) Summary and output stage.

<!-- Media -->

In this work, we propose a self-evolving framework for multi-agent medical consultation (SeM-Agents) based on large language models. This framework incorporates various doctor roles and auxiliary roles. Patients present with their medical problems and background, and a Primary Care Doctor assigns the most appropriate specialist doctor agents based on the specific condition of the patient. The roles of Radiologist, Pathologist, and Pharmacist are essential and always included. Specialist doctor agents engage in multi-round discussions to share information, and once a consensus is reached, the consultation results are filtered through a Safety and Ethics Reviewer before the outcomes and recommendations are issued. Depending on their accuracy, the results are saved in different databases for reference in future consultations. Our contributions include: 1. We have introduced a dynamically expandable medical consultation framework that adapts to the patient's condition. 2. The framework evolves its consultation capabilities using two experience databases as the number of consultations increases. 3. The framework adopts an efficient residual discussion structure, enabling agents to efficiently access content from earlier shared speech pools, avoid information redundancy and contamination, and ensure the clarity and relevance of medical consultations.

## II. METHOD

In this section, we will introduce the details of our medical consultation framework, where we have arranged multiple roles (Primary Care Doctor, General Internal Medicine Doctor, General Surgeon, Pediatrician, Obstetrician and Gynecologist, Radiologist, Neurologist, Pathologist, Pharmacist, Chain-of-Thought Reviewer, Safety and Ethics Reviewer, Patient). This arrangement enhances the framework's universality, enabling it to effectively address a wide range of complex medical scenarios. Fig. 1 shows the overview of the framework we propose, which is divided into three critical stages: (A) Assignment, (B) Consultation, and (C) Summary.

## A. Aranging specialist doctors

When the patient Agent comes for a consultation carrying personal background $C$ and medical problem $Q$ ,the Primary Care Doctor Agent assigns specialist doctor Agents based on the specific circumstances of the patient. To ensure the triage doctor's output is more accurate and structured, we configure the Primary Care Doctor Agent with a few-shot example as a reference. The workflow is as follows: This description can be expressed by the following formula:

$$
\text{ Roles } = \operatorname{LLM}\left( {\text{ Agents } \mid  C,Q}\right) . \tag{1}
$$

$$
\text{Roles} \subseteq  \left\{  {{\text{Agent}}_{1},{\text{Agent}}_{2},\ldots ,{\text{Agent}}_{n}}\right\}  \text{.} \tag{2}
$$

Where roles, apart from Radiologist, Pathologist, and Pharmacist, are chosen based on the specific circumstances, avoiding information pollution by too many unrelated expert agents.

## B. Multi-round consultations

Once the specialist doctors for the consultation have been determined, the consultation process begins. In the initial round of consultation, each specialist doctor presents their views based on the patient's condition and provides an option ID and content for the issue at hand. At this stage, each specialist doctor agent cannot observe the others' remarks. All comments from this round are stored in a shared speech pool $\left( {S}_{1}\right)$ . Once all the specialist doctors have concluded,the consultation moves to the next round of remarks.

<!-- Meanless: Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on October 27,2025 at 02:45:21 UTC from IEEE Xplore. Restrictions apply.-->




Starting from the second round, each specialist doctor agent can access remarks stored in the shared speech pool from the previous round. They integrate these insights to optimize prompts,formulating their own responses denoted as ${S}_{2,k}$ (where ${S}_{2,k}$ represents the response of the $k$ -th specialist doctor in the second round), and specify an option ID along with content relevant to the current issue for further discussion and decision-making.

From the $i + 1$ round $\left( {i \geq  2}\right)$ ,specialist doctor agents can review the remarks from rounds $i$ and $i - 1$ in the shared speech pool. Incorporating the collective remarks from the previous two rounds to enhance the prompt, they articulate their own views and provide an option ID and content for the issue. The discussion continues until all the expert doctor agents reach a consensus on the answers. If consensus is not reached or the number of discussion rounds is below the maximum (set at 10 ), the discussion continues. If the maximum rounds are reached without achieving consensus, the decision is made by majority rule; if votes are evenly distributed, a final answer is randomly selected from the Agents' choices. This residual discussion mode reduces information pollution and enhances the efficiency of the discussion, while also reducing memory size. Furthermore, each expert doctor agent can access deeper layers of memory, which helps prevent any single expert doctor agent from being overly influenced by other agents, thereby mitigating the occurrence of hallucinations to some extent. The consultation process is defined according to Algorithm 1.

<!-- Media -->

Algorithm 1 Multi-Round Medical Consultation Process

---

Initialize: speech pool ${S}_{1} = \left\{  {{s}_{1,1},{s}_{1,2},\ldots ,{s}_{1,n}}\right\}$
Compute for Round 2:
for each specialist $k$ do
	${S}_{2,k} \leftarrow  f\left( {{S}_{1},C,Q}\right)$
end for
Subsequent Rounds:
Set $i = 2$
while not Consensus $\left( {S}_{n}\right)$ and $i \leq$ MaxRounds do
	for each specialist $k$ do
		${S}_{i + 1,k} \leftarrow  g\left( {{S}_{i},{S}_{i - 1},C,Q}\right)$
	end for
	Increment $i$
	Consensus check:
	if $\forall k,m \in  \{ 1,\ldots ,n\}  : {S}_{n,k} = {S}_{n,m}$ then
		Set Consensus $\left( {S}_{n}\right)  =$ True
	end if
end while

---

<!-- Media -->

## C. Summary and output stage

During this phase,the final output(C)is subjected to a review by the Safety and Ethics Reviewer Agent, who filters and refines the consultation conclusions, identifying any unsafe aspects and finalizing the conclusions(R). These conclusions are then compared with the correct outcomes. If the consultation conclusions are accurate, the patient's background(B),the problem,and the discussions from the final consultation round are archived in the Correct Answer Knowledge Base (CorrectKB). Conversely, if the consultation results in incorrect conclusions, the session is abstracted by the Chain-of-Thought Reviewer. This abstraction includes the patient's background and problem, structured according to the initial hypotheses, analysis process, final conclusion, and reasons for error, and is then stored in the Chain of Thought Knowledge Base (ChainKB).

When the next patient arrives, the background and problem of the patient are used to retrieve the most similar cases from the two databases via cosine similarity, thus enhancing the prompts(P)for the specialist doctor agents. To preserve the independent reasoning of each specialist doctor agent, references to the two knowledge bases are generally not made before the initial round of discussions. Instead, these references are utilized starting from the second round, particularly when divergent opinions arise. However, if a consensus emerges in the first round, the bases may be consulted post-discussion as a reflective measure. This process is given in Algorithm 2.

<!-- Media -->

Algorithm 2 Summarization and Enhancement of Prompts

---

Input: $C,B$ ,CorrectKB,ChainKB
Review and Validate:
$R \leftarrow  \operatorname{Review}\left( C\right)$
$D \leftarrow  \left\{  \begin{array}{ll} \text{ Correct }\mathrm{{KB}} & \text{ if }\operatorname{Valid}(R \\  \text{ Chain }\mathrm{{KB}} & \text{ otherwise } \end{array}\right.$
Knowledge Management:
if $D =$ CorrectKB then
	Store(R, CorrectKB)
else
	abstraction $\leftarrow  \operatorname{Abstract}\left( R\right)$
	Store(abstraction, ChainKB)
end if
Consultation Enhancement:
similarity $\leftarrow$ CosineSim(B,CorrectKB,ChainKB)
$P \leftarrow$ Retrieve(similarity,CorrectKB,ChainKB)
enhancedPrompt $\leftarrow$ Enhance(P)
Apply Enhanced Prompt:
round $\leftarrow  1\; \vartriangleright$ Initialize round count
while not Consensus(C)do
	UsePrompt(enhancedPrompt, round)
	round $\leftarrow$ round +1
	ifround $= 1$ and $\operatorname{Consensus}\left( C\right)$ then
		ConsultKBs $\left( P\right) \; \vartriangleright$ Reflective measure post-discussion
	end if
end while

---

<!-- Media -->

## III. EXPERIENCE

## A. Datasets

We use the MedQA [26] and PubMedQA [27] datasets to validate our framework. The MedQA dataset consists of USMLE-style questions, each offering four or five possible answers, designed to assess medical knowledge and practical skills. PubMedQA, based on research paper abstracts, presents questions with Yes/No/Maybe answers, aiming to evaluate the performance of natural language processing models in academic question answering. The final results are all measured on the test set of each dataset. The Correct Answer Knowledge Base and the Chain of Thought Knowledge Base only include experiences from the training set of each dataset.

<!-- Meanless: Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on October 27,2025 at 02:45:21 UTC from IEEE Xplore. Restrictions apply.-->




## B. Main Results

In this experimental section, we primarily explore the Zero-shot accuracy advantages of our proposed framework, SeM-Agents, in the medical consultation domain, and validate the contributions of each component within our approach. For this subset of experiments, all agents in our framework utilize gpt-4-turbo, with SeM-Agents undergoing 600 consultation rounds-a benchmark chosen after considering both performance and cost. Overall performance can be referred to in Table I, where the foundational model used across all configurations is gpt-4-turbo. 'Single-Agent' denotes performance using only gpt-4-turb as our baseline. 'Single-Agent (w/) CoT' incorporates a 'Let's think step by step' approach in the answering process. '1 Round Multi-Agent' employs a single-round voting mechanism following the majority rule principle. '10 Rounds Multi-Agent-sequential speaking' describes a scenario where specialist doctor agents speak in a sequential order after listening to the previous agent's opinion, a method that generally underperforms compared to the voting model in medical fields. Although our method shows lower accuracy on the MedQA dataset compared to Medprompt [8]-likely due to Medprompt only being tested in four-option scenarios-it achieves higher accuracy on the PubQA dataset and on average than Medprompt. Table I illustrates that the discussion modes and experiential growth proposed positively impact overall performance, with an interesting observation that correct experiences contribute more significantly to accuracy improvements than abstracted CoT experiences, which is intuitively consistent.

<!-- Media -->

TABLE I

MAIN RESULTS ON ACCURACY ACROSS MEDQA AND PUBMEDQA DATASETS

<table><tr><td>Method</td><td>$\mathbf{{MedQA}}\left( \% \right)$</td><td>$\mathbf{{PubMedQA}\left( \% \right) }$</td><td>Average(%)</td></tr><tr><td>Single-Agent</td><td>77.4</td><td>75.3</td><td>76.4</td></tr><tr><td>Single-Agent (w/) CoT</td><td>76.6</td><td>76.9</td><td>76.8</td></tr><tr><td>Medprompt [8]</td><td>90.2</td><td>82.0</td><td>86.1</td></tr><tr><td>1 Round Multi-Agent</td><td>78.2</td><td>73.7</td><td>76.0</td></tr><tr><td>10 Rounds Multi-Agent</td><td>78.5</td><td>74.0</td><td>76.3</td></tr><tr><td>10 Rounds Multi-Agent sequential speaking</td><td>77.8</td><td>72.9</td><td>75.4</td></tr><tr><td>MedAgents [18]</td><td>83.7</td><td>76.8</td><td>80.3</td></tr><tr><td>SeM-Agents (w/o) residual discussion mode (ours)</td><td>88.6</td><td>79.3</td><td>83.95</td></tr><tr><td>SeM-Agents (w/o) bases</td><td>83.3</td><td>77.1</td><td>80.2</td></tr><tr><td>SeM-Agents (w/o) correct answer knowledge base (ours)</td><td>86.2</td><td>80.7</td><td>83.5</td></tr><tr><td>SeM-Agents (w/o) CoT knowledge base (ours)</td><td>89.0</td><td>82.5</td><td>85.8</td></tr><tr><td>SeM-Agents (ours)</td><td>89.2</td><td>83.1</td><td>86.2</td></tr></table>

<!-- Media -->

## C. Self-Evolving Experiment

The self-evolving, growth-capable framework of multi-agent doctors, which continuously improves through consultation experiences, often aligns more closely with practical requirements than static frameworks—an intuitively appealing notion. Herein, we demonstrate how the number of consultation cases and the volume of stored case experiences influence accuracy variations on test sets across two datasets, MedQA and PubMedQA. Each dataset contributes half of the cases in two experience databases: the Correct Answer Knowledge Base and the Chain of Thought Knowledge Base. We conduct tests using foundation models gpt-3.5-turbo and gpt-4-turbo. As illustrated in Fig. 2, the overall trend shows a gradual increase in accuracy as the number of consultation samples grows (with a slight decline observed around 100 cases), and tends to plateau after reaching 600 cases.

<!-- Media -->

<!-- figureText: 0.90 gpt-4-turbo gpt-3.5-turbc 300 (a) gpt-3.5-turbc 300 600 Consultation Case (b) 0.85 Accuracy 0.80 0.75 0.70 0.66 0.84 0.82 0.80 Accuracy 0.74 -->

<img src="https://cdn.noedgeai.com/bo_d3vdvik601uc738lu9u0_3.jpg?x=1049&y=352&w=468&h=600&r=0"/>

Fig. 2. Accuracy Variations: (a) Tested on MedQA (b) Tested on PubMedQA

<!-- Media -->

## D. Impact of Different Foundation Models

We demonstrate the accuracy of different foundational models on two datasets, each enhanced with knowledge bases from 600 consultation cases. Table II shows that gpt-4-turbo remains the optimal foundational model for the SeM-Agents framework. Meanwhile, other foundational models also demonstrate excellent performance within the SeM-Agents framework, suggesting its adaptability across different foundational models.

<!-- Media -->

TABLE II

ACCURACY COMPARISONS ACROSS MEDQA AND PUBMEDQA USING DIFFERENT FOUNDATION MODELS

<table><tr><td>Backbone</td><td>MedQA(%)</td><td>PubMedQA(%)</td><td>Average</td></tr><tr><td>LLaMA3-8B [28]</td><td>70.1</td><td>64.9</td><td>67.5</td></tr><tr><td>GLM4 [29]</td><td>74.3</td><td>75.1</td><td>74.7</td></tr><tr><td>DeepSeek-v2 [30]</td><td>75.7</td><td>74.5</td><td>75.1</td></tr><tr><td>gpt-3.5-turbo</td><td>75.2</td><td>76.6</td><td>75.9</td></tr><tr><td>gpt-4-turbo</td><td>89.2</td><td>83.1</td><td>86.2</td></tr></table>

<!-- Media -->

## IV. CONCLUSION

In this paper, we introduce a novel multi-agent framework for medical consultation that employs a residual discussion mode to reduce information pollution and enhance discussion efficiency. By leveraging two experience databases, this framework dynamically improves overall consultation accuracy. However, the overall performance of the framework largely depends on the capabilities of the foundational model used to store and utilize consultation experiences, which may limit its performance. Despite these limitations, our approach still excels in current medical consultation scenarios. REFERENCES

<!-- Meanless: Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on October 27,2025 at 02:45:21 UTC from IEEE Xplore. Restrictions apply.-->




[1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., "Gpt-4 technical report," arXiv preprint arXiv:2303.08774, 2023.

[2] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar et al., "Llama: Open and efficient foundation language models," arXiv preprint arXiv:2302.13971, 2023.

[3] J. Wang, L. Chen, A. Khare, A. Raju, P. Dheram, D. He, M. Wu, A. Stol-cke, and V. Ravichandran, "Turn-taking and backchannel prediction with acoustic and large language model fusion," in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 12121-12125.

[4] I. Malkiel, U. Alon, Y. Yehuda, S. Keren, O. Barkan, R. Ronen, and N. Koenigstein, "Segllm: Topic-oriented call segmentation via llm-based conversation synthesis," in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 11361-11365.

[5] F. Chi, Y. Wang, P. Nasiopoulos, and V. C. Leung, "Multi-modal gpt-4 aided action planning and reasoning for self-driving vehicles," in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 7325-7329.

[6] X. Zhang, C. Tian, X. Yang, L. Chen, Z. Li, and L. R. Petzold, "Alpacare: Instruction-tuned large language models for medical application," arXiv preprint arXiv:2310.14558, 2023.

[7] Z. Bao, W. Chen, S. Xiao, K. Ren, J. Wu, C. Zhong, J. Peng, X. Huang, and Z. Wei, "Disc-medllm: Bridging general large language models and real-world medical consultation," arXiv preprint arXiv:2308.14346, 2023.

[8] H. Nori, Y. T. Lee, S. Zhang, D. Carignan, R. Edgar, N. Fusi, N. King, J. Larson, Y. Li, W. Liu et al., "Can generalist foundation models outcompete special-purpose tuning? case study in medicine," Medicine, vol. 84, no. 88.3, pp. 77-3, 2023.

[9] H. Ye, T. Liu, A. Zhang, W. Hua, and W. Jia, "Cognitive mirage: A review of hallucinations in large language models," arXiv preprint arXiv:2309.06794, 2023.

[10] A. Pal, L. K. Umapathi, and M. Sankarasubbu, "Med-HALT: Medical domain hallucination test for large language models," in Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL). Singapore: Association for Computational Linguistics, Dec. 2023, pp. 314-334.

[11] R. Liu, R. Yang, C. Jia, G. Zhang, D. Yang, and S. Vosoughi, "Training socially aligned language models on simulated social interactions," in The Twelfth International Conference on Learning Representations, 2024.

[12] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, "Improving factuality and reasoning in language models through multiagent debate," in Forty-first International Conference on Machine Learning, 2024.

[13] C. Qian, W. Liu, H. Liu, N. Chen, Y. Dang, J. Li, C. Yang, W. Chen, Y. Su, X. Cong, J. Xu, D. Li, Z. Liu, and M. Sun, "ChatDev: Communicative agents for software development," in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Aug. 2024, pp. 15174-15186.

[14] J. S. Park, J. O'Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, "Generative agents: Interactive simulacra of human behavior," in Proceedings of the 36th annual acm symposium on user interface software and technology, 2023, pp. 1-22.

[15] Z. Du, C. Qian, W. Liu, Z. Xie, Y. Wang, Y. Dang, W. Chen, and C. Yang, "Multi-agent software development through cross-team collaboration," arXiv preprint arXiv:2406.08979, 2024.

[16] C. Qian, Z. Xie, Y. Wang, W. Liu, Y. Dang, Z. Du, W. Chen, C. Yang, Z. Liu, and M. Sun, "Scaling large-language-model-based multi-agent collaboration," arXiv preprint arXiv:2406.07155, 2024.

[17] S. Hong, M. Zhuge, J. Chen, X. Zheng, Y. Cheng, J. Wang, C. Zhang, Z. Wang, S. K. S. Yau, Z. Lin, L. Zhou, C. Ran, L. Xiao, C. Wu, and J. Schmidhuber, "MetaGPT: Meta programming for a multi-agent collaborative framework," in The Twelfth International Conference on Learning Representations, 2024.

[18] X. Tang, A. Zou, Z. Zhang, Z. Li, Y. Zhao, X. Zhang, A. Cohan, and M. Gerstein, "Medagents: Large language models as collaborators for zero-shot medical reasoning," in ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024.

[19] L. Chen, J. Q. Davis, B. Hanin, P. Bailis, I. Stoica, M. Zaharia, and J. Zou, "Are more llm calls all you need? towards scaling laws of compound inference systems," arXiv preprint arXiv:2403.02419, 2024.

[20] W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang, "Memorybank: Enhancing large language models with long-term memory," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 17, 2024, pp. 19724-19731.

[21] A. Zhao, D. Huang, Q. Xu, M. Lin, Y.-J. Liu, and G. Huang, "Expel: Llm agents are experiential learners," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 17, 2024, pp. 19632-19642.

[22] C. Qian, Y. Dang, J. Li, W. Liu, Z. Xie, Y. Wang, W. Chen, C. Yang, X. Cong, X. Che, Z. Liu, and M. Sun, "Experiential co-learning of software-developing agents," in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Aug. 2024, pp. 5628-5640.

[23] C. Qian, J. Li, Y. Dang, W. Liu, Y. Wang, Z. Xie, W. Chen, C. Yang, Y. Zhang, Z. Liu et al., "Iterative experience refinement of software-developing agents," arXiv preprint arXiv:2405.04219, 2024.

[24] S. Jiang, Y. Wang, and Y. Wang, "Selfevolve: A code evolution framework via large language models," arXiv preprint arXiv:2306.02907, 2023.

[25] J. Li, S. Wang, M. Zhang, W. Li, Y. Lai, X. Kang, W. Ma, and Y. Liu, "Agent hospital: A simulacrum of hospital with evolvable medical agents," arXiv preprint arXiv:2405.02957, 2024.

[26] D. Jin, E. Pan, N. Oufattole, W.-H. Weng, H. Fang, and P. Szolovits, "What disease does this patient have? a large-scale open domain question answering dataset from medical exams," Applied Sciences, vol. 11, no. 14, p. 6421, 2021.

[27] Q. Jin, B. Dhingra, Z. Liu, W. Cohen, and X. Lu, "Pubmedqa: A dataset for biomedical research question answering," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 2567-2577.

[28] A. . M. Llama Team, "The llama 3 herd of models," arXiv preprint arXiv:2407.21783, 2024.

[29] J. Zeng, B. Zhang, Y. Ma, K. Sun, H. Zhou, Y. Liu et al., "Chatglm: A family of large language models from glm-130b to glm-4 all tools," arXiv preprint arXiv:2406.12793, 2024.

[30] A. Liu, B. Feng, B. Wang, B. Wang, B. Liu, C. Zhao et al., "Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model," arXiv preprint arXiv:2405.04434, 2024.

<!-- Meanless: Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on October 27,2025 at 02:45:21 UTC from IEEE Xplore. Restrictions apply.-->

