# CogAgent: A Visual Language Model for GUI Agents

Appendix

<!-- Media -->

## 1. Details of Training Configurations

We report the detailed training settings of CogAgent in Table 7, and model configurations of CogAgent in Table 8.

<table><tr><td>Configurations</td><td>Pre-train</td><td>Multi-task</td></tr><tr><td>Total steps</td><td>60,000</td><td>10,000</td></tr><tr><td>Warmup steps</td><td>500</td><td>500</td></tr><tr><td>Batch size</td><td>4,608</td><td>1,024</td></tr><tr><td>Learning rate</td><td colspan="2">$2 \times  {10}^{-5}$</td></tr><tr><td>Learning rate decay</td><td colspan="2">Cosine</td></tr><tr><td>Weight decay</td><td colspan="2">0.05</td></tr><tr><td>Dropout ratio</td><td colspan="2">0.1</td></tr><tr><td>Adam $\epsilon$</td><td colspan="2">$1 \times  {10}^{-5}$</td></tr><tr><td>Adam $\beta$</td><td colspan="2">(0.9, 0.95)</td></tr></table>

Table 7. Training settings of pre-training and multi-task fine-tuning.

<table><tr><td colspan="2">VLM decoder</td></tr><tr><td>Architecture</td><td>Vicuna-1.5-7B + visual expert</td></tr><tr><td>Layers</td><td>32</td></tr><tr><td>Hidden size</td><td>4,096</td></tr><tr><td>Attention heads</td><td>32</td></tr><tr><td colspan="2">Low-resolution visual encoder</td></tr><tr><td>Architecture</td><td>EVA2-CLIP-E</td></tr><tr><td>Input resolution</td><td>${224} \times  {224}$</td></tr><tr><td>Patch size</td><td>${14} \times  {14}$</td></tr><tr><td colspan="2">High-resolution visual encoder</td></tr><tr><td>Visual encoder</td><td>EVA2-CLIP-L</td></tr><tr><td>Input resolution</td><td>1120×1120</td></tr><tr><td>Patch size</td><td>${14} \times  {14}$</td></tr><tr><td colspan="2">Cross Attention</td></tr><tr><td>Hidden size</td><td>1,024</td></tr><tr><td>Attention heads</td><td>32</td></tr></table>

Table 8. Model configurations of CogAgent.

<!-- Media -->

## 2. Details of Evaluation Datasets

In this section, we will provide a detailed overview of the datasets used in our evaluations .

### 2.1. General VQA

- VQAv2 [1]. VQAv2 is designed for visual question answering with natural images, covering a wide range of question types including yes/no, numerical counting, and more open-ended inquiries. The dataset comprised of a collection exceeding 200,000 images, paired with more than 1.1 million questions. Each question is paired with 10 answers annotated by different annotators.

- OK-VQA [23]. OK-VQA (Outside Knowledge Visual Question Answering) dataset is constructed to evaluate visual question-answering skills that require external knowledge. Models need to combine image content and common sense to answer questions. The dataset includes 14,055 open-ended questions, each accompanied by 5 ground truth answers.

- MM-Vet [41]. MM-Vet is designed to evaluate the overall capability of generalist visual language models in a zero-shot manner. It integrates 6 core VL capabilities to solve complex tasks (including recognition, OCR, knowledge, language generation, spatial awareness, and math), and explores 16 distinct integrations resulting from the combination of these capabilities. As for evaluation metrics, it utilizes a language model-based evaluator tailored for open-ended responses.

- POPE [19]. POPE (Polling-based Object Probing Evaluation) is a dataset constructed to assess the object hallucination problem in large visual language models. It employs a polling-based object probing method, transforming hallucination assessment into a binary classification challenge. This is achieved by prompting large vision-language models (LVLMs) with straightforward Yes-or-No queries regarding the objects in question (for example, "Is there a car in the image?"). Our evaluation is conducted under the dataset's most challenging setting: the adversarial setting.

### 2.2. Text-rich VQA

- OCR-VQA [27]. OCR-VQA dataset comprises 207,572 images of book covers, paired with more than 1 million question-answer pairs. The questions inquire about book information including title, edition, year, author, and genre of the book, which requires text recognition and comprehension abilities.

- TextVQA [34]. TextVQA is a benchmark of visual reasoning based on text in images. Models need to


<!-- Media -->

<table><tr><td>Task</td><td>Dataset</td><td>Description</td><td>Split</td><td>Metrics</td></tr><tr><td rowspan="4">General VQA</td><td>VQAv2</td><td>VQA on natural images.</td><td>test-dev</td><td>VQA Score(↑)</td></tr><tr><td>OK-VQA</td><td>VQA on natural images requiring outside knowledge.</td><td>val</td><td>VQA Score (↑)</td></tr><tr><td>MM-Vet</td><td>Conversational style VQA on integrated capabilities.</td><td>test</td><td>GPT-4 score(↑)</td></tr><tr><td>POPE</td><td>VQA for hallucination assessment. The adversarial setting is used.</td><td>test</td><td>F1 score(↑)</td></tr><tr><td rowspan="6">Text-rich VQA</td><td>OCR-VQA</td><td>VQA on images of book covers.</td><td>test</td><td>EM (↑)</td></tr><tr><td>TextVQA</td><td>VQA on natural images containing text.</td><td>test</td><td>VQA Score (↑)</td></tr><tr><td>ST-VOA</td><td>VQA on natural images requiring textual understanding.</td><td>test</td><td>ANLS (↑)</td></tr><tr><td>ChartQA</td><td>VQA about charts with visual and logical reasoning.</td><td>test</td><td>VQA Score (↑)</td></tr><tr><td>InfoVQA</td><td>VQA on infographics.</td><td>test</td><td>ANLS (↑)</td></tr><tr><td>DocVQA</td><td>VQA on document images.</td><td>test</td><td>ANLS (↑)</td></tr><tr><td rowspan="2">GUI Agent</td><td>Mind2Web</td><td>Web behavior prediction given snapshots and historical actions.</td><td>test</td><td>step SR (↑)</td></tr><tr><td>AITW</td><td>Android behavior prediction given snapshots and historical actions.</td><td>test</td><td>Matching Score (↑)</td></tr></table>

Table 9. Summary of the evaluation benchmarks.

<!-- Media -->

incorporate the textual information in the images and reason over it to answer TextVQA questions. It comprises a total of 28,408 images and 45,336 questions.

- ST-VQA [4]. ST-VQA is designed to emphasize the significance of exploiting the semantic information present within images in textual form during the VQA process. It comprises tasks of diverse difficulties, for which recognizing the scene text and performing necessary reasoning is required to generate the answer. The dataset comprises 23,038 images sourced from multiple public datasets and 31,791 question-answer pairs.

- ChartQA [24]. ChartQA is a benchmark of question-answering about logical and visual reasoning on charts. It consists of 20,882 charts curated from four different online sources, 9,608 manual written questions, as well as 23,111 questions automatically generated with T5 according to human-written chart summaries.

- InfographicVQA(InfoVQA) [26]. The task of In-foVQA is to answer questions centering on a given in-fographic image. The answers to most questions can be extracted from the given documents, while the answers to a small percentage of questions are not extractive. There are 5K Images collected from the Internet and 30K manually annotated questions in the dataset.

- DocVQA [25]. DocVQA focuses on question-answering given a document image. The answer for questions is often a span of text from the given documents. There are ${12}\mathrm{\;K}$ images and ${50}\mathrm{\;K}$ manually annotated questions in the datasets.

#### 2.3.GUI Agent

CogAgent is evaluated on two GUI agent datasets, Mind2Web and Android in the Wild (AITW), corresponding to computer agent and smartphone agent respectively.

- Mind2Web [10]. Mind2Web is designed to develop and evaluate web agents capable of executing intricate tasks on various websites based on language directions. While existing datasets for web agents commonly rely on simulated or overly simplified web data, Mind2Web utilizes real-world websites and is annotated by human annotators. It gathers data from 137 websites covering 31 domains, and collects over 2,000 open-ended tasks, each accompanied by a crowd-sourced action sequence.

In mind2web, evaluated agents are asked to accomplish a designated task on a chosen website by performing a sequence of actions. Each instance (i.e. a specific task) in Mind2Web contains a task description, action sequence, and webpage snapshots. Each action in the sequence is a (Target element, Operation) pair, and Operation includes Click, Type (with additional value), and Select (with additional value). Each action is paired with the concurrent webpage snapshots in a variety of formats including raw HTML code, DOM tree, screenshot, etc. As for CogAgent, we choose screenshot images as the input representation of web-sites; as for other language-based agents, HTML is chosen as the input representation.

Following Deng et al. [10] and Zeng et al. [42], we formalize the problem as: first choose the target webpage element among top-k (k=10 or 50) candidates, then predict specific operations. The top-k candidates are provided by the candidate generation model in Deng et al. [10]. Step success rate (step SR) is reported on 3 out-of-domain test sets (cross-website, cross-domain,


cross-task) as metric. Only predictions with the same target element and operation as the ground truth are regarded as correct.

- Android in the Wild (AITW) [31]. AITW is constructed to develop and evaluate Android device-control systems that are capable of understanding and acting upon human natural language instructions by directly manipulating the device's user interface. This dataset significantly surpasses others in its category in terms of size, encompassing 715k episodes across 30k distinct instructions, and covering four Android versions (v10-13). It also includes eight types of devices, ranging from Pixel 2 XL to Pixel 6, each with different screen resolutions. AITW consists of five subsets: GoogleApps, Install, WebShopping, General, and Single. The subsets have distinct tasks, while are in the same data format. Each episode (i.e. action sequence) comprises three components: a goal instruction provided in natural language, a user action sequence, and a corresponding screenshots sequence.

As for screenshots, AITW only provides screenshot images and does not provide tree-based representations of UI. This is because a large portion of them in Smartphone applications are of low quality or even do not exist, and adopting the tree-based representation would strongly limit agents' applications. For visual agents, screenshots are provided to the agents in image format; for language-model-based agents evaluated by Zhan and Zhang [43], the textual representations of OCR and icons formatted in HTML syntax are provided. As for actions, AITW considers a variety of action types including tapping, swiping, typing, going home, going back, entering, etc. For each action, models are required to predict the exact action type; for tap, swipe and type, models are further required to predict the position, direction, and content to be typed, respectively. The detailed standard for computing matching scores is provided in Rawles et al. [31].

## 3. Derivation of Acceleration for High- Resolution Cross-Module

Suppose that ${L}_{{I}_{\mathrm{{lo}}}},{L}_{{I}_{\mathrm{{hi}}}}$ and ${L}_{T}$ are the lengths of the low-resolution image, high-resolution image and text sequences. Let ${H}_{\text{cross }},{H}_{\text{dec }}$ be the number of attention heads in cross-attention and self-attention,and ${d}_{\text{cross }},{d}_{\text{dec }}$ be the dimension of each attention head.

If using our high-resolution cross-module, the computational complexity of attention is

$$
\begin{aligned} {\mathrm{T}}_{\text{improved }} = \mathbf{O}\left( {\left( {{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}}\right) {L}_{{I}_{\mathrm{{hi}}}}{H}_{\text{cross }}{d}_{\text{cross }}}\right. &  \\ \left. {+{\left( {L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}\right) }^{2}{H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}}\right) \text{.} &  \end{aligned} \tag{5}
$$

If not utilizing the high-resolution cross-module and directly substituting low-resolution images with high-resolution ones, the computational complexity would be

$$
{\mathrm{T}}_{\text{original }} = \mathbf{O}\left( {{\left( {L}_{{I}_{\mathrm{{hi}}}} + {L}_{T}\right) }^{2}{H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}}\right) . \tag{6}
$$

The reduction factor of the computational complexity in attention, ${\mathrm{T}}_{\text{original }}/{\mathrm{T}}_{\text{improved }}$ ,equals to

$$
\frac{{\left( {L}_{{I}_{\mathrm{{hi}}}} + {L}_{T}\right) }^{2}{H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}}{\left( {{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}}\right) {L}_{{I}_{\mathrm{{hi}}}}{H}_{\mathrm{{cross}}}{d}_{\mathrm{{cross}}} + {\left( {L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}\right) }^{2}{H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}} \tag{7}
$$

$$
 = \frac{{L}_{{I}_{\mathrm{{hi}}}} + {L}_{T}}{{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}}\frac{\left( {{L}_{{I}_{\mathrm{{hi}}}} + {L}_{T}}\right) {H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}}{{L}_{{I}_{\mathrm{{hi}}}}{H}_{\text{cross }}{d}_{\text{cross }} + \left( {{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}}\right) {H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}} \tag{8}
$$

$$
 = \frac{{L}_{{I}_{\mathrm{{hi}}}} + {L}_{T}}{{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}}\frac{\left( {{L}_{{I}_{\mathrm{{hi}}}} + {L}_{T}}\right) \frac{{H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}}{{H}_{\mathrm{{cross}}}{d}_{\mathrm{{cross}}}}}{{L}_{{I}_{\mathrm{{hi}}}} + \left( {{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}}\right) \frac{{H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}}{{H}_{\mathrm{{cross}}}{d}_{\mathrm{{cross}}}}} \tag{9}
$$

- Case 1: ${L}_{{I}_{\mathrm{{lo}}}},{L}_{T} \ll  {L}_{{I}_{\mathrm{{hi}}}}$ .

Given that ${L}_{{I}_{\mathrm{{lo}}}}$ is much smaller than ${L}_{{I}_{\mathrm{{hi}}}}$ ,when ${L}_{T}$ also satisfies being much smaller than ${L}_{{I}_{\mathrm{{hi}}}}$ ,both $\frac{{L}_{{I}_{\mathrm{{lo}}}}}{{L}_{{I}_{\mathrm{{hi}}}}}$ and $\frac{{L}_{T}}{{L}_{{I}_{\mathrm{{hi}}}}}$ become first-order small quantities. If conducting a 0-th order approximation for the complexity reduction factor, we obtain:

$$
\frac{{\mathrm{T}}_{\text{original }}}{{\mathrm{T}}_{\text{improved }}} = \frac{{L}_{{I}_{\mathrm{{hi}}}}\left( {1 + \frac{{L}_{T}}{{L}_{{I}_{\mathrm{{hi}}}}}}\right) }{{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}}\frac{{L}_{{I}_{\mathrm{{hi}}}}\left( {1 + \frac{{L}_{T}}{{L}_{{I}_{\mathrm{{hi}}}}}}\right) \frac{{H}_{\mathrm{{dec}}}}{{H}_{\text{cross }}}{d}_{\text{cross }}}{{L}_{{I}_{\mathrm{{hi}}}}\left( {1 + \frac{{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}}{{L}_{{I}_{\mathrm{{hi}}}}}\frac{{H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}}{{H}_{\text{cross }}}}\right) }
$$

(10)

$$
 = \frac{{L}_{{I}_{\mathrm{{hi}}}}\left( {1 + \frac{{L}_{T}}{{L}_{{I}_{\mathrm{{hi}}}}}}\right) }{{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}}\frac{\left( {1 + \frac{{L}_{T}}{{L}_{{I}_{\mathrm{{hi}}}}}}\right) \frac{{H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}}{{H}_{\mathrm{{cross}}}{d}_{\mathrm{{cross}}}}}{1 + \frac{{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}}{{L}_{{I}_{\mathrm{{hi}}}}}\frac{{H}_{\mathrm{{dec}}}{d}_{\mathrm{{dec}}}}{{H}_{\mathrm{{cross}}}{d}_{\mathrm{{cross}}}}}
$$

(11)

$$
 \approx  \frac{{L}_{{I}_{\mathrm{{hi}}}}}{{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}} \tag{12}
$$

- Case 2: Our implementation.

In our implementation, ${d}_{\text{cross }} = {32},{H}_{\text{cross }} = {32}$ ,and we inherits ${d}_{\text{dec }} = {128},{H}_{\text{dec }} = {32}$ from CogVLM- 17B. Both high- and low-resolution encoders patchify images with ${14} \times  {14}$ -pixel patches,thus ${L}_{{I}_{\mathrm{{hi}}}} = {6400}$ , ${L}_{{I}_{\text{lo }}} = {256}$ . Subsitute the values to Eq. (9),we obtain

$$
\frac{{\mathrm{T}}_{\text{original }}}{{\mathrm{T}}_{\text{improved }}} = \frac{{6400} + {L}_{T}}{{256} + {L}_{T}}\frac{4\left( {{6400} + {L}_{T}}\right) }{{6400} + 4\left( {{256} + {L}_{T}}\right) } \tag{13}
$$

$$
 > \frac{{6400} + {L}_{T}}{{256} + {L}_{T}} \tag{14}
$$

That is to say,our method leads to at least $\frac{{L}_{{I}_{\mathrm{{hi}}}} + {L}_{T}}{{L}_{{I}_{\mathrm{{lo}}}} + {L}_{T}} = \; \frac{{6400} + {L}_{T}}{{256} + {L}_{T}} \times$ acceleration which is a stringent lower bound. Taking pre-training as an example, with ${L}_{T}$ typically below 512 in earlier stages,our high-resolution cross-module can get an over ${25} \times$ reduction in computational cost.


## 4. Performance Analysis on AITW

Android in the Wild (AITW) dataset is a large-scale dataset for Android device agents. Our analysis centers on instances where the CogAgent's responses diverge from the annotated ground truth, focusing on these discrepancies.

In our analysis, we conducted a sampling and manual verification of cases where there was a discrepancy between the model's responses and the labeled answers. We sampled instances and discovered that in 42% of these, the model's response actually represented an alternative correct method, rather than an error. This phenomenon is common due to the multiplicity of ways to accomplish a function on a smart-phone. For example, many phones are equipped with both Google's app and a Google search bar, and searches can be conducted using either. During the use of smartphones, human users may exhibit simultaneous needs for both in-depth exploration and extensive browsing. For example, when browsing news, CogAgent might identify relevant results on the current page and select them immediately, while some users may prefer to scroll through the entire page first. Furthermore, the data collection for annotations might involve redundant or overly detailed steps, whereas CogAgent can directly find the most efficient path. Fig. 4 illustrates these typical examples of multiple correct answers.

Given the nature of smartphone usage, wherein accomplishing a task often involves multiple viable paths, it is advisable for future studies to explore a virtual environment with an evaluation of the accuracy of task completion following a sequence of actions, aligning more closely with real-world human usage scenarios.

<!-- Media -->

<!-- figureText: User:<br>User:<br>Open a new Chrome incognito tab<br>Set an alarm for $3\mathrm{{pm}}$ .<br>CogAgent:<br>CogAgent: tap [[0.512, 0.403]].<br>3.45<br>tap [[0.672, 0.784]]<br>Ground Truth:<br>Ground Truth:<br><br>User:<br>User:<br>What's the news in Brazil?<br>What's the price of the Galaxy phone on eBay?<br>CogAgent:<br>CogAgent:<br>tap [[0.318, 0.246]].<br>tap [[0.434, 0.190]].<br>Ground Truth:<br>Ground Truth:<br>scroll down.<br>scroll right. -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_3.jpg?x=140&y=1360&w=709&h=519&r=0"/>

Figure 4. Instances in the AITW dataset where CogAgent provides accurate responses that differ from the annotated ground truth.

<!-- Media -->

## 5. Samples of Pre-train Data

Samples of pre-train data are listed in this section, including data for text recognition (Fig. 5), visual grounding (Fig. 6), and webpage-html pairs (Fig. 7).

has kept nine players that featured in last year's FIBA Basketball World Cup. While 36-year-old Eduardo Mingas is set to play his sixth consecuti African Championship, Roberto Fortes who has featured for Angola in consecutive World Championships, will make his AfroBasket debut in Tuni. from 19-30 August. Fortes stood out throughout the training camp, havin helped Angola beat Venezuela in

being about a half hour from the city's center. I was struck by ho lovely it was, until I learned one could n't eat the fish, an it asn't a good idea to do to to much swimming radiation being a prob Halladay no-hitter came in the first possesson start of his care Roy Halladay of the Phillies pitched the second no-hitter in Mi League Baseball postseason history on Wednesday as the Philadelp Phillies beat the Cincinnati Reds 4-0 in their National League

<!-- Media -->

<!-- figureText: ${g}^{\prime }\left( y\right) : = \frac{f\left( x\right) }{\sqrt{{\widetilde{d}}_{\lambda ,h}\left( x\right) }} - \frac{f\left( y\right) }{\sqrt{{\widetilde{d}}_{\lambda ,h}\left( y\right) }}\;{L}^{p} \rightarrow {L}_{\left( {{2n} - 1}\right) /{p}^{\prime }}^{p}$<br>$\gamma = {6.96} \times {10}^{8}\mathrm{\;m}$<br>${\kappa }^{2} = - \frac{1}{2}{\nabla }^{\mu }{\chi }^{\nu }{\nabla }_{\mu }{\chi }_{\nu }$<br>On Charge Quantization and Abelian Gauge<br>Horizontal Symmetries<br>L.N. Epele ${}^{1}$ , C.A. Garcia Canal ${}^{1}$ , and Wilson A. Passer<br>Sequester 23, 2023<br>Abstract<br>using the total above only for a top quark mass at two boot, we have<br>2. If $\mathcal{O} \subset \mathcal{O}$ and $P \in \mathcal{A}\left( \mathcal{O}\right)$ is a projector then $P{B}_{\mathcal{O}} = P{B}_{\mathcal{O}}$<br>linear quantum mechanics is now sufficiently clear that one may bed that<br>brought forth against it up to now.<br>Acknowledgment -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_3.jpg?x=903&y=1006&w=712&h=930&r=0"/>

Figure 5. Samples of pre-train data for text recognition.


<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_4.jpg?x=208&y=272&w=569&h=385&r=0"/>

A bride [[592,179,866,956]] and groom [[717,160,862,882]] standing near the water [[425,426,990,898]] at sunset.

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_4.jpg?x=289&y=758&w=406&h=459&r=0"/>

Chicken and rice dish [[301,160,997,792]] in a pan [[282,107,998,839]], with spoons [[091,476,630,953]] and plate [[002,004,258,374]] next to it.

<!-- figureText: A bluebird $\left\lbrack \left\lbrack {{302},{413},{640},{752}}\right\rbrack \right\rbrack$ sitting on a branch coffee mugs $\left\lbrack \left\lbrack {{279},{588},{677},{804}}\right\rbrack \right\rbrack$ . -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_4.jpg?x=228&y=1344&w=507&h=567&r=0"/>

<!-- Media -->

Figure 6. Samples of pre-train data for visual grounding. Snowmobiling Daytime - Excl. Transport\\n </a>\\n </h2>\\n

<!-- Media -->

<!-- figureText: ICE DOMES<br>ACTIVITIES ACCOMMODATION PACKAGES<br>B things to do<br>Snowmobiling Daytime - Excl. Transport TELEMAN<br>To dive "The scracter of the north" in a snowy weldsmans playground.<br>Snowmobiling Daytime - Incl. Transport<br>Headman<br>From 2145 NOK<br>Snowmobiling Evening - Excl. Transport BARDON -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_4.jpg?x=908&y=320&w=693&h=403&r=0"/>

[Box] [[30, 281, 870, 445]]

[HTML]<div>\\n<div>\\n <div>\\n <h2>\\n <!-- -->\\n <a>\\n <a>\\n <!---->\\n </a>\\n <div>\\n <div>\\n <!--->\\n <span>\\n Tromsø\\n </span>\\n <span>\\n <a>\\n <span>\\n </span>\\n <b>\\n Map\\n </b>\\n <b>\\n Map\\n </b>\\n </a>\\n </span>\\n </div>\\n </div>\\n </ div>\\n </div>\\n <a>\\n <div>\\n <p>\\n Enjoy an unforgettable snowmobile safari through magical snow-covered mountains, dramatic landscapes and scenic valleys. Learn how to drive "the scooter of the north" in a snowy wilderness playground.\\n </p>\\n <div>\\n <span>\\n Read more\\n </span>\\n <span>\\n <span>\\n </span>\\n </span>\\n </div>\\n </div>\\n </ad>\\n<!-- >\\n</div>

<!-- figureText: Home Photo Biog About S. 19. Bus Buy Prints Goilery Contact Me<br>2013- MY YEAR IN PHOTO BLOGGING<br>you really should as there are some great features including the Tableland feature where<br>Tags: 2012 Comparison, 2013, 2013 in Review, Astronomy Bioggerg Brian May, Dr.<br>Brian May For, Guitarini, ISS: ISS over Waterford, photo blog. Photo Biogging, Photo at<br>Fox, photoking Queen Guitarist, Queen Guitarist Brian May, Review, Hendew of<br>the Yiar, Shaine McDonald, Virtual Astroecomer, Year in Review -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_4.jpg?x=903&y=1251&w=701&h=415&r=0"/>

[Box] [[190, 510.203125, 1088.21875, 682.703125]] [HTML] <p>\\n WordPress via the Jetpack plugin publish reports on your blogs which are very interesting. If you have a WordPress blog and you have not yet installed Jetpack then you really should as there are some great features including the 'Publicize' feature where you can auto-update your Facebook, Twitter and Google+ accounts with your posts. My [...]\\n</p>

Figure 7. Samples of webpage-html pairs.

<!-- Media -->


## 6. Details of Fine-Tuning Data

### 6.1. Human annotation

To enable CogAgent to function as an agent on various apps and websites, we assembled a team of over ten annotators to collect and label screenshots. To ensure the quality of the annotations and prevent overly simplistic, homogenized content, the annotation process was divided into two phases.

In the first phase, annotators were required to independently select from a pool of apps and websites, and capture screenshots of pages strongly relevant to the main function of that app/website, with no less than 20 screenshots for each app/website. Subsequently, for each screenshot, the annotators would label them according to the following aspects:

1. Five buttons. List the names of five clickable buttons from the screenshots. The types of buttons should be diverse.

2. Three clickable areas. List three clickable areas from the screenshots, e.g. text boxes, search boxes, click-able images, hyperlinks, etc.

3. Two questions extracting information from the image. Ask questions involving the textual information in the images. For instance, "In what year did the user in the image register?".

4. One operation requirement. What function of the webpage/app would you use in this interface? For example, adding the comment: "cool!". The requirement shouldn't be vague such as "adding comment".

In the second phase, annotators are required to provide grounding annotation for the questions and operational methods proposed in the first phase (the aforementioned parts 3 and 4). For example, for Figure 8, a possible annotation would be:

1. Buttons: Back; Search; Subscribed; Home; Library.

2. Clickable areas: Avatar in the middle top; Video preview in the middle bottom; Personal profile.

3. Question 1: Based on the page, how many followers does this author have?

Answer: According to the personal profile at the top [[013,568,802,188]], this author has 4.97M followers.

Question 2: Based on the page, how many videos has this author posted?

Answer: According to the personal profile at the top [[013,568,802,188]], this author has posted 502 videos.

<!-- Media -->

<!-- figureText: 13:32<br>山（ша<br>Key & Peele<br>5<br>NOOICE<br>Key & Peele<br>@KeyAndPeele 4.97M subscribers 502 videos<br>The Key & Peele channel is the place to watch comedy sketches from Keegan-Michael Key and Jordan Peele, whose Comedy Ce...<br>paramountplus.com and 4 more links<br>A Subscribed $\checkmark$<br>Home<br>Videos<br>Shorts<br>Live<br>Playlists<br>Co<br>BEST OF<br>15:33<br>Key & Peele's Most Unpredictable Bosses<br>Key & Peele : 451K views : 2 months ago<br>A<br>8 -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_5.jpg?x=1000&y=214&w=514&h=1054&r=0"/>

Figure 8. Samples of human-collected screenshot.

<!-- Media -->

4. Operation requirement: Based on the page, write out the steps to complete the following function: Follow this author.

Answer: Click on Subscribed [[049,826,728,078]] to follow this author.

### 6.2. Conversion of Agent Datasets

To convert Mind2Web to natural language with GPT4, we use the following prompt:

---

<!-- Footnote -->

Imagine that you are a robot operating a computer. Like how humans operate the computer, you can move the mouse, click with the mouse, or type some texts with the keyboard.

**Your ultimate task is: "Find the lowest-priced round trip flight with hotel on May 2 from Kathmandu, Nepal KTM to Shanghai, China PVG and return on May 5. Book a double room and check out with the default flights.".**

<!-- Footnote -->

---


You are given previous actions: (format: element $\rightarrow$ operation)

1. [link] Flight + Hotel $\rightarrow$ CLICK,

2. [textbox] Where from? $\rightarrow$ TYPE: KATHMANDU,

3. [div] Tribhuvan Intl Airport (KTM), Nepal $\rightarrow$ CLICK,

4. [textbox] Where to? $\rightarrow$ TYPE: SHANGHAI,

5. [div] Pudong Intl Airport (PVG), China → CLICK,

6. [span] Sat 8 Apr - Sat 15 Apr $\rightarrow$ CLICK,

7. [checkbox] 2 May 2023 → CLICK,

8. [checkbox] 5 May 2023 → CLICK,

Furthermore, you **secretly** know the following steps of action. **(But you have to pretend this is the result of your derivation and you don't know it before)**

1. [button] Search $\rightarrow$ CLICK,

2. [button] Recommended $\rightarrow$ CLICK,

3. [div] Lowest Price $\rightarrow$ CLICK,

4. [svg] $\rightarrow$ CLICK,

5. [button] Choose your room $\rightarrow$ CLICK,

6. [button] Book Business Double Room A → CLICK,

7. [svg] $\rightarrow$ CLICK,

8. [svg] $\rightarrow$ CLICK,

9. [button] Choose $\rightarrow$ CLICK

Please formulate an operational guide for future operations for solving the task. The guide includes:

1. Plan: A **multi-step future** plan that does not involve low-level operations **(start from current screen and action, DON'T include previous actions)**; steps indexed by numbers. **Be sure to pretend that you don't know the future interface and actions, and don't know the elements not existing on the current screen.**

2. Action: The specific **immediate** action that needs to be taken (Corresponding to the** very next action** [button] Search $\rightarrow$ CLICK at the box \{"x_left": 0.876, "y_left": 0.308, "width": 0.063, "height": 0.034\}), which should be detailed enough to directly control the operation, such as specifying the exact location (or box) to click, the exact content of the text to type, the item to select.

**Output Format:** A JSON dictionary strictly following the format:

\{ "plan": "...<Your Plan Here>", "action": "...<Your Action Here>", "operation": "[button] Search → CLICK at the box "x_left": 0.876, "y_left": 0.308, "width": 0.063, "height": 0.034"\} **Please do not output any content other than the JSON format. Don't mention elements that only appear in HTML such as svg, div.**

The returned answer from GPT-4 would be:

\{"plan": "1. After searching, you'll see a list of flight and hotel packages. 2. Review the recommended options, usually highlighted for convenience. 3. Focus on finding the lowest priced option. You might need to navigate through various sorting or filtering features to do this. 4. Once you've found the lowest priced option, select it to view more details. 5. In the detailed view, you'll have to choose your room type. Look for options that fit your requirements, like a business double room. 6. After selecting the room, proceed to finalize your booking. This might involve confirming your choice and entering any required information. 7. Review all the details carefully before finalizing the booking to ensure everything is correct.",

"action": "Click the 'Search' button to proceed with the search based on the entered criteria. This button is located towards the right side of the screen.",

"operation": "[button] Search $\rightarrow$ CLICK at the box \{"x_left": 0.876, "y_left": 0.308, "width": 0.063, "height": 0.03\}" \}

## 7. Failure cases

The failure cases can be summarized into four categories (Fig. 9):

1. Incorrect action prediction;

2. Incorrect coordinate prediction;

3. Incorrect GUI observation;

4. Hallucination.

<!-- Media -->

<!-- figureText: Type 1. Incorrect action prediction. (Left)<br>Task: Play Groove Is In the Heart.<br>Action: Touch the play button to start playing 'Groove Is In the Heart'.<br>Grounded Operation: tap [[780,862]]<br>Type 2. Incorrect coordinate prediction. (Right)<br>Task: Change item 4 of Category A in the table to 11.<br><br>Action: Touch the cell corresponding to Item 4 of Category A. Grounded Operation: tap [[627,820]]<br>Type 3. Incorrect GUI observation. (Left)<br>15-48<br>KEEK<br>Task: Turn on Low Power Mode.<br>Action: Touch the area that corresponds to the 'Low Power Mode' toggle switch.<br>Grounded Operation: tap [[842,215]]<br>Type 4. GUI hallucination. (Right)<br>Los Angeles<br>Task: Search for the way to Las Vegas.<br>Action: Tap on the search bar at the top of the screen, which<br>may currently contain the word 'Chico'.<br>Grounded Operation: tap [[804,159]] -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_6.jpg?x=902&y=1390&w=714&h=462&r=0"/>

Figure 9. Failure cases of GUI Agent.

<!-- Media -->

## 8. More Generated Samples of CogAgent

To further demonstrate the capabilities of CogAgent, we provide additional generated samples in this section.


<!-- Meanless: User: What's the proper procedure for "Book a room of the lowest price in New York on Dec 1st"? (with grounding)<br>est price in New York on Dec 1st"? (with grounding) -->

<!-- Media -->

<!-- figureText: 1<br><br>Find your next stay<br>Search deals on hotels, homes, and much more..<br>Black Friday Deals<br>Save 30% or more on stays<br>Offers -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_7.jpg?x=227&y=424&w=614&h=400&r=0"/>

Plan: 1. Type 'New York' into the search box to specify the destination.

2. Select the correct New York location from the auto-suggested list.

3. Adjust the check-in and check-out dates to December 1st, respectively.

4. Execute the search by clicking the 'Search' button.

5. Once the results are displayed, sort them by price, selecting the lowest first.

6. Review the options and select the room offer that is the lowest priced.

7. Proceed to book the selected room following the website's booking procedure.

Next Action: Type 'New York' into the textbox located at the coordinates \{'x_min': 0.175, 'y_min': 0.309, 'width': 0.28, 'height': 0.047\} on the screen.

Grounded Operation: [textbox] Where are you going? -> TYPE: New York at the box [[175,309,455,356]]

<!-- figureText: 3<br>Booking.com<br>(A Steps) Trights % Fight 4 model A Carnentals & Attractions . Subport task<br>Find your next stay<br>Search deals on hotels, homes, and much more..<br>23. New York<br>[C] Check-in Date - Check-out Date<br>A. 2 adults - D children - 1 room<br>Black Friday Deals<br>November 2023<br>December 2023<br>a Mo Th We Th Se<br>Siii Mo This Th Fe Se<br>___<br>1 2<br>皮质黄色，白色白<br>3 4 5 6 7 8 9<br>Offer ends in 4 days<br>Save 30% or more or Book before Nov 29, 2023 - stay an end of 2024<br>16 分别 25 分232426<br>$\begin{array}{lllllll} {17} & {18} & {19} & {20} & {21} & {22} & {23} \end{array}$<br>$\begin{array}{lllll} {26} & {27} & {20} & {29} & {30} \end{array}$<br>24 25 26 27 28 29 30 -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_7.jpg?x=221&y=1171&w=621&h=401&r=0"/>

Plan: 1. Confirm the checkout date by clicking on the appropriate date, which in this case is 1 December 2023.

2. Initiate the search for hotels by clicking the 'Search' button to find available rooms.

3. Once the results are displayed, sort the list of hotels by price, selecting the option to order them from the lowest to highest price.

4. Review the sorted list and select the hotel offering the lowest price.

5. Proceed to book the selected hotel room following the website's booking procedure.

Next Action: Click on the calendar box corresponding to the date 1 December 2023 to set it as the checkout date.

Grounded Operation:[gridcell] 1 December 2023 -> CLICK at the box [[739,564,776,611]]

<!-- figureText: 2<br><br>Find your next stay<br>Search deals on hotels, homes, and much more...<br>ANGWOOK<br>Black Friday Deals<br>Save 30% or more on stays<br>Offers<br>Work, relax, or do a bit of both -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_7.jpg?x=901&y=412&w=626&h=414&r=0"/>

Plan: To book the lowest priced room in New York on Dec 1st, the next steps would involve:

1. Adjusting the check-in and check-out dates to December 1st, respectively.

2. Executing a search for available rooms.

3. Sorting the search results by price, from low to high.

4. Selecting the room offer with the lowest price.

5. Proceeding to book the chosen room.

Next Action: Click on the 'Check-in Date' textbox at the specified coordinates to bring up the date selection interface.

Grounded Operation:[textbox] Check-in Date $\rightarrow$ CLICK at the box [[438,309,549,357]]

<!-- figureText: 4<br>Booking.com<br>Regular Spain<br>(四川) D. Flights Q. Flight Holdi A. Currentals Q. Arrections 至 Airportradis<br>Find your next stay<br>Search deals on hotels, homes, and much more...<br>Bioms York<br>例 Fri, Dec 1 - Set, Dec 2<br>A 2 adults - 0 children - 1 room<br>I'm traveling for work<br>Black Friday Deals<br>Save 30% or more on stays<br>Book before Nov 29, 2023 - stay arguments before the<br>end of 2024 -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_7.jpg?x=908&y=1171&w=619&h=401&r=0"/>

Plan: 1. Click the Search button to initiate the search with the current criteria.

2. Once the search results are displayed, sort the listings by price, selecting the option to order them from low to high.

3. Review the sorted list and select the room offer that is the lowest in price.

4. Click on the 'Book' button for the chosen offer to proceed with the reservation process.

Next Action: Click the Search button located at the bottom right of the search parameters box to execute the search for hotels in New York from December 1st to December 2nd for 2 adults and 0 children in 1 room.

Grounded Operation:[button] Search -> CLICK at the box [[798,367,875,424]]

<!-- Meanless: -d samples of CogAgent.<br>Figure 10. More generated samples of CogAgent. -->


<!-- Meanless: User: What's the proper procedure for "Book a room of the lowest price in New York on Dec 1st"? (with grounding) -->

<!-- figureText: 5<br>Booking.com<br><br><br><br><br>New York: 279 properties found<br>1. Surfay: Our Top Meta ::<br>✘<br>SS 959<br>Very Good in -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_8.jpg?x=242&y=297&w=575&h=378&r=0"/>

Plan: 1. Click on the 'Sort by: Our Top Picks' button to change the sorting criteria.

2. Select 'Price (lowest first)' from the sort options to arrange the listings by price in

3. Identify the lowest-priced hotel listed after the sort operation.

4. Click on the 'See availability' button for the lowest-priced hotel to view available rooms.

5. Choose a room option that fits the requirements, such as a specific price range or room type.

6. Proceed with booking by clicking 'Reserve' and following the subsequent booking steps. Next Action: Click at the box with coordinates starting approximately at 29.8% from the left of the screen and 27.2% from the top of the screen.

Grounded Operation:[button] Sort by:Our Top Picks -> CLICK at the box [[298,272,420,307]]

<!-- Media -->

7

<!-- Media -->

<!-- figureText: 25 max) to require<br>Separtment<br>[10] Set, Dec 1 - Set, Dec 1<br>New York: 279 properties found<br>riner by:<br>Jazz on the Park Youth Hostel $= 1$ -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_8.jpg?x=251&y=974&w=554&h=369&r=0"/>

Plan: 1. Click on the 'Jazz on the Park Youth Hostel' to view more details about the property and to verify it meets all booking requirements including availability for the specified date, price, and amenities.

2. Review the detailed information provided after the click to ensure it matches the desired criteria.

3. If the property is suitable, proceed to reserve a room by following the booking instructions which may include selecting a room type, providing personal details, and making a payment.

Next Action: Move the cursor over the box that represents the 'Jazz on the Park Youth Hostel' link with coordinates starting at approximately 38.9% from the left of the screen and 38.6% from the top of the screen.

Grounded Operation:[link] Jazz on the Park Youth Hostel Opener West Side, New Yor... -> CLICK at the box [[389,386,810,435]]

<!-- figureText: 9<br>Availability<br>Boules Match<br>#FLOOG1 - Sat.Dec2<br>Speed Batheren Troom<br>with Shared Bathman<br>Oney Troom left on our airs<br>2 twin bolds<br>ESSION<br>Stripedest<br>SS 513<br>DomitonsRoam<br>1 [解析] local<br>di Berlin dorm<br>Oney than only<br>Guest reviews -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_8.jpg?x=239&y=1648&w=568&h=381&r=0"/>

<!-- Media -->

Figure 11. More generated samples of CogAgent.

<!-- Media -->

<!-- figureText: 6<br>Booking.com<br>[23] have fort<br>New York: 279 properties found<br>at Sorting: Our Rap Picks 6<br>or unavailable for your dates on our site.<br>The FIDI Hotal $+ + + + \square$<br>Very Good [12]<br>S\$ 959 -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_8.jpg?x=924&y=290&w=590&h=387&r=0"/>

Plan: 1. Click on the 'Price (lowest first)' option to sort the properties by price in ascending order.

2. Review the sorted list and identify the property with the lowest price.

3. Select the property by clicking on it.

4. Once on the property's page, choose a room by clicking on the 'See availability' button for that room.

5. Follow the booking process by selecting the date, room type, and any other necessary options, then proceed to reserve the room by clicking 'Reserve'.

Next Action: Click on the 'Price (lowest first)' sorting option located towards the top of the webpage, within the specified coordinates on the screen.

Grounded Operation:[li] Price (lowest first) -> CLICK at the box [[288,379,494,414]]

<!-- figureText: 8<br>Search<br>Jazz on the Park Youth Hostel<br>Q. New York.<br>2.6 Health spaces $\frac{2\pi }{3}$ free WDF, the WDF $\frac{2\pi }{3 + 2}$ do have front desk $\frac{2\pi }{3 + 2}$ freedom [22] treating<br>5. Air conditioning $\widetilde{\Delta }$ Bagange storage $\widetilde{\Delta }$ Designated smoking area -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_8.jpg?x=944&y=966&w=563&h=375&r=0"/>

Plan: After the current action, the next steps should include confirming the reservation details such as dates, number of guests, and room preferences. Then proceed to enter personal information required for booking. Following that, review all the details for accuracy, choose a payment method, and complete the booking process. Finally, wait for a confirmation message or email from the hotel to ensure the reservation has been successfully made.

Next Action: Click on the 'Reserve' button located towards the right side of the screen in a box with coordinates \{'x_min': 0.781, 'y_min': 0.724, 'width': 0.106, 'height': 0.038\}.

[[781,724,887,762]]

<!-- Media -->

Plan: 1. Click on the 'Select amount' button for the room with the lowest price listed. 2. Choose the desired payment option, such as 'Pay in advance' if available and offered at a lower rate.

3. Confirm the selection and proceed to booking by clicking on 'Continue to book'.

4. Complete the booking process by entering personal and payment details as required.

5. Review the reservation summary and finalize the booking by confirming all details are correct and clicking 'Book now'.

Next Action: Click on the 'Select amount' button located towards the right side of the scr een for the accommodation option with the lowest price displayed, which is the 'Standard

Grounded Operation: [button] Select amount -> CLICK at the box [[739,272,906,318]]


<!-- Media -->

User: Could you guide me through the steps to

"Write an email to cvpr2024@gmail.com, with subject of 'I'm CogAgent', and the main body of 'Here is CogAgent sending the email.' " ?

<!-- figureText: 1<br>Outlook<br><br>Disks 32<br>Resummed theory<br>O =<br>OpenAI Incident - Elevated error rates on DALLE API and Labs + 15<br>November 2023<br>(c)<br>23 Archive<br>(i) Getting too much small? Unsubscribe | Manage subscription:<br>Add total<br>0<br>E. S. Translate message to: | Never translate from English<br>to be a constant<br>52 Inbox 52<br>CB link C 2<br>SOpenAI<br>Drain<br>(c)<br>>> Sentire<br>⓺<br>13 Arctive<br>Elevated error rates on DALLE API and Labs<br>17 Notes<br>6<br>15 comes.<br>Oracle $n$ .<br>(c)<br>New incident: Investigating<br>48. Get Outlook with<br>Now 15, DR-18 PST<br>(C) ${AP}$ (C) Labs -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_9.jpg?x=235&y=395&w=616&h=370&r=0"/>

<!-- figureText: 2<br><br>(2) Bec -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_9.jpg?x=908&y=388&w=627&h=380&r=0"/>

## CogAgent:

CogAgent:

Plan: 1. Click on the 'New email' button to start a new email. Plan: 1. Type the email address into the 'To' field.

2. Enter the recipient's email address in the 'To:' field. 2. Enter the subject line in the 'Subject' field.

3. Enter the subject in the 'Subject:' field. 3. Use the keyboard to type the main body of the email.

Next Action: Move the cursor to the 'To' textbox, click on it to focus on it, and type 'cvpr2024@gmail.com'. 4. Type the main body of the email in the 'Body' text area.

Next Action: Click on the 'New mail' button located at the top left corner of the screen, within the Outlook interface.

<!-- figureText: Apters<br>Send Y<br>To [ typr2024@gmail.com ×<br>CE-BO<br>I'm CogAgent<br>Draft amount at 16:19 -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_9.jpg?x=915&y=981&w=620&h=370&r=0"/>

3 的，因为8是 $x = 0$ 的最大值的 $\theta  = 1$ ， $\theta  = \theta$ ， $\theta  = 1/2$

Send

To expr2024@gmail.com × It is

Dreft assemble to 10.16

## CogAgent:

## CogAgent:

Plan: 1. Click on the text area to bring up the keyboard if it's not already active. Plan: 1. Type the subject line into the 'Subject' field.

2. Type the main body of the email, 'Here is CogAgent sending the email.'

2. Enter the main body of the email in the large text area below the subject line.

3. Use the 'Send' button to send the email.

Next Action: Click on the 'Subject' textbox, which is currently highlighted and type 'I'm CogAgent'. Then, use the keyboard to type the rest of the message in the large text area below it. 3. Find and click the 'Send' button to send the email.

Next Action: Move the cursor to the text area where the message content needs to be typed and type 'Here is CogAgent sending the email.'

<!-- figureText: 5<br>Insert<br>Format text<br>Draw<br>Options<br>Aptos<br>12<br>-<br>To Cupr2024@gmail.com $\times$<br>Ca Bac<br>I'm CogAgent<br>Here is CogAgent sending the email. -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_9.jpg?x=227&y=1597&w=627&h=387&r=0"/>

Figure 12. More generated samples of CogAgent.

<!-- Media -->


User: Could you guide me through the steps to "Listen to the music named You Raise Me Up." ?

<!-- Media -->

<!-- figureText: 1<br>2<br>15:41<br>115.61<br>IS:32<br>... 56<br>网<br>Search<br>Mail<br>CogAgent:<br>Q. Artists, Songs, Lyrics, and More<br>CogAgent:<br>Plan: 1. Locate the Music app icon.<br>Plan: 1. Tap on the search bar at the top of the screen.<br>2. Open the app to find the music library.<br>Music<br>2. Enter the song title into the search field.<br>3. Search for 'You Raise Me Up' by using the search function within the app.<br>Get 1 month of free music.<br>3. Execute the search by pressing the search button or enter key on the keyboard.<br>Next Action: Move the cursor to the search bar and click it to activate the text entry mode for typing in the song name 'You Raise Me Up'.<br>Next Action: Click on the Music app icon located in the second row from the bottom, third column from the left.<br>Plus your entire music library on all your devices. 1 month free then ¥5.00/month.<br>Tryli Free<br>Browse Categories<br>asserth<br>Not Playing<br>A<br>O<br>O<br>(10)<br>园<br>Q<br>4<br>3<br>15:31<br>11 56<br>Q. Artists, Songs, Lyrics, and...<br>Cancel<br>Q You Raise Me Up in Appl<br>Apple Music<br>Your Library<br>op Re<br>Artists<br>Songs<br>Playl<br>...<br>Jason Zhang<br>You Raise Me Up<br>><br>CogAgent:<br>Song - Westlife<br>Plan: 1. Identify the search bar at the top of the screen. 2. Review the list of songs that appear as a result of the search query.<br>Miss Pretty Song - Allen Su<br>...<br>Plan: 1. Tap on the search bar at the top of the screen. 2. Enter the text 'You Raise Me Up' into the search field.<br>Westlife<br>><br>CogAgent:<br>Artist<br>You Raise Me Up<br>...<br>Song - Sarah Alainn<br>Allen Su<br>><br>Artist<br>Next Action: Look at the search results section just below the search bar to find the song 'You Raise Me Up'.<br>网<br>Song - Lana Del Rey<br>...<br>3. Execute the search by pressing the search button or enter key on the keyboard.<br>Song - Martin Hurkens<br>You Raise Me Up Song - Secret Garden<br>Love Story (Taylor's Version)<br>q<br>W<br>t<br>у<br>0<br>p<br>Next Action: Click on the search bar where it says 'Artists, Songs, Lyrics, and...', then type 'You Raise Me Up'.<br>Westlife: Chill<br>><br>Playlist - Westlife: Chill<br>You Raise Me Up<br>Song - Josh Groban<br>a<br>S<br>d<br>h<br>佥<br>C<br>b<br>n<br>123<br><br>space<br>Not Playing<br>>><br>典<br>也<br><br>((+.2))<br>园<br>Q -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_10.jpg?x=194&y=434&w=1351&h=1432&r=0"/>

<!-- Media -->

Figure 13. More generated samples of CogAgent.


<!-- Media -->

<!-- figureText: 开<br>白云边菜饭<br>国<br>& Go to Merusea Village<br>284m<br>3 Teleport Waypoint<br>R -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_11.jpg?x=304&y=218&w=1114&h=517&r=0"/>

User: What's shown in this picture? Describe in detail.

CogAgent: The screenshot shows a character standing in the game Genshin Impact, looking at a teleport waypoint. The HUD displays various elements: on the top left is a mini-map with icons indicating different locations; on the top right are character portraits and health bars for Raiden Shogun, Bennett, and Kaedehara Kazuha, along with an UID number; below these is a quest prompt 'Go to Merusea Village' with a distance indicator of ${284}\mathrm{\;m}$ ; in the center is the glowing Teleport Waypoint; at the bottom are the character's level (Lv. 90),health points (25123/3461), stamina (623), and other game UI elements like food, a key, and a compass.

User: I'm looking for guidance on how to "Go to the teleport waypoint"?

CogAgent: Plan: 1. Identify the direction to the teleport waypoint from the current location on the map.

2. Navigate through the environment or use in-game transportation methods to reach the destination. 3. Confirm arrival at the teleport waypoint and prepare for the next step of the journey within Genshin Impact.

Next Action: Move the cursor over the 'Teleport Waypoint' icon located towards the center right of the screen, just above the character's feet, and left-click to activate it.

Grounded Operation: [generic] Teleport Waypoint -> CLICK at the box [[506,497,686,547]]

User: Who are the teammates on the current mission?

CogAgent: The teammates on the current mission are Raiden Shogun, Bennett, and Kaedehara Kazuha.

User: I'm looking for guidance on how to "Find out how to view a detailed map"?

CogAgent: Plan: 1. Identify the map icon on the screen.

2. Click on the map icon to open the detailed map view.

Next Action: Move the cursor over the top left corner of the screen and click on the compass-like icon with 'N' inside it.

<!-- figureText: 22:13<br><br>FIRST AND LAST<br>LINES FROM<br>11:10<br>The First & Last lines Spoken In Friends | Netflix<br>Still Watching Netflix $\cdot {3.6}\mathrm{M}$ views $\cdot 4$ years ago<br>SMELLYCAT<br>Taylor Swift & Lisa Kudrow - Smelly Cat (Live on The 1989 World Tour)<br>Taylor Swift's Tour Extras - 1.1M views - 2 years<br><br>8<br><br><br>Library -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_11.jpg?x=283&y=1261&w=388&h=776&r=0"/>

User: When did the Smelly Cat video post?

CogAgent: The Smelly Cat video post was made 2 years ago.

User: How long is the Smelly Cat video?

CogAgent: The Smelly Cat video is 3 minutes and 23 seconds long.

User: What should I do if I want to watch the Smelly Cat video (in the middle of the screen) and share it with my friend Rena?

CogAgent: Plan: 1. Play the Smelly Cat video by Taylor Swift & Lisa Kudrow to ensure it is fully loaded and ready for viewing.

2. Locate the share icon on the screen, which is typically represented by a symbol such as a paper airplane or an arrow pointing outwards from a box.

3. Tap the share icon to open sharing options.

4. Find and select the option to send via messaging app, like WhatsApp, to share the video with friends.

5. In the messaging app, locate the contact named Rena.

6. Send the video link to Rena through the messaging app.

Next Action: Touch the area in the middle of the screen where the 'Smelly Cat' thumbnail is displayed, corresponding to [Live on The 1989 World Tour] by Taylor Swift & Lisa Kudrow - Smelly Cat. This action is intended to start playing the video.

Grounded Operation: tap [[701,498]]

Figure 14. More generated samples of CogAgent.


<!-- figureText: 14:46<br>!!!! 令 JOI<br>Post<br>Disney<br>@Disney<br>So many iconic Disney and Pixar films were released in November! - Take this quiz to find out how much you know about them.<br>05:02 - 2023/11/15 from Earth - 117K Views<br>$\textbf{46}$ Reposts2Quotes319Likes13Bookmarks<br>Q<br>い<br>0<br>闪<br>土<br>Disney & @Disney - 1d<br>What animal does Robin Hood disguise himself as when competing in the tournament of the golden arrow?<br>Duck<br>8%<br>Pig<br>4%<br>Stork<br>61%<br>Wolf<br>27%<br>1,156 votes - Final results<br>0.1<br>1213<br>0.36<br>由3.4K<br>贝　众<br>Disney @@Disney ·1d<br>Postyour reply<br>囧<br>Q<br>品<br>Q<br>☑<br>14:33 如图 1<br><br>Community post<br>Dogs<br>Harvey &<br>Follow<br>@harveyyross<br>True?<br>Some people will never understand how much someone can love a dog, but that's okay the dog knows..<br>6:44 · 07 Nov 23 · 10.8K Views<br>17 Quotes 241 Likes 6 Bookmarks<br>t2<br>口<br>90.0%<br>*Farm *Girl* * @far... ·19h Replying to @harveyyross Awww, that's so sweet. And yes,<br>$\equiv$<br>< -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_12.jpg?x=286&y=316&w=367&h=1603&r=0"/>

<!-- Media -->

User: Describe the image in detail.

CogAgent: The image is a screenshot of Twitter posts from the official Disney account. In the first post, there's an emoji-filled statement that reads 'So many iconic Disney and Pixar films were released in November! Take this quiz to find out how much you know about them.' This tweet was posted on 2023/11/15 at 05:02 and has received 46 reposts, 2 quotes, 319 likes, and 13 bookmarks. Below it, another tweet states 'What animal does Robin Hood disguise himself as when competing in the tournament of the golden arrow?' with multiple choice options like Duck, Pig, Stork, Wolf.

User: Which option has the highest number of votes?

CogAgent: The 'Stork' option has the highest number of votes with 61%.

User: How can I "Follow Disney's Twitter account"?

CogAgent: Plan: 1. Locate the 'Follow' button on the screen. 2. Click the 'Follow' button to follow Disney's Twitter account.

Next Action: Move the cursor over the 'Follow' button next to the Disney logo and click it.

User: Describe the image in detail.

CogAgent: The image showcases a tweet from an account named 'Harvey' with the handle '@harveyyross'. The tweet contains an illustration of a young girl hugging a large dog. Above this, there is another text that reads: 'Some people will never understand how much someone can love a dog, but that's okay the dog knows...'. Below the illustration, there are statistics showing the tweet has been viewed 10.8K times, liked by 241 users, retweeted 6 times, and bookmarked by 6 users. Additionally, there is a reply to this tweet can 1 from an account named 'Farm Girl' with the handle '@far...', which says: 'Awww, that's so sweet. And yes,' indicating agreement o timesr appreciation for the original message.

User: Does the tweet have any reply? What is it?

CogAgent: Yes, there is a reply to the tweet from an account named 'Farm Girl' with the handle '@far...'. The reply reads: 'Awww, that's so sweet. And yes,' indicating agreement or appreciation for the original message.

Figure 15. More generated samples of CogAgent.


<!-- Media -->

<!-- figureText: 11:15<br>Notes<br>iTunes Store<br>Books<br>Home<br>Q Search<br>2,052<br>22:35<br>!!!! ? 83<br>Meetings<br>C<br>①<br>n<br>8<br>圈<br>10<br>New Meeting<br>Join<br>Schedule<br>Share Screen<br>Add a calendar<br>Today<br>Zoom user's Zoom Meeting Meeting ID: 8741305 9500<br>Start -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_13.jpg?x=139&y=263&w=393&h=1272&r=0"/>

<!-- Media -->

User: Describe the screen in detail.

CogAgent: The picture shows a mobile phone screen with various icons arranged in a grid. There are 24 app icons on the screen, including three cute cartoon bears at the bottom of the screen. These bears have different expressions and colors. In addition to the app icons, there is also a digital clock displaying "11:15" at the top left corner of the screen.

User: What apps do I have?

CogAgent: The apps on the phone are divided into two categories: system apps and third-party apps. System apps include "FaceTime", "Calendar", "Photos", "Mail", "Clock", "Maps", "Weather", "Notes", "Reminders", "Stocks", "Videos", "iTunes Store", "App Store", "Books", "Health", "Home", "Music", and "Moji Weather". Third-party apps include "Q Search" and other unidentified apps, which cannot be identified from their icons alone.

User: What should I do if I want to see photos taken 2 days ago?

CogAgent: You should tap on the "Photos" app, and it will display all your photos. If you want to see photos taken 2 days ago, you can select a date from the options at the top of the screen or use the search function in the Photos app to find specific photos.

User: If I want to take a photo, what should I do?

CogAgent: You should tap on the "Camera" app to take a photo.

User: What should I do if I want to call Susan?

CogAgent: You should tap on the "Phone" app to call Susan.

User: How can I know Susan's number?

CogAgent: You can call Susan by searching for her number in the phone's contact list or on your phone's address book. If you have saved her number, you can directly dial it from there.

User: What should I do if I want to download youtube?

CogAgent: You should tap on the "App Store" app, and then search for YouTube in it. Once you find the official YouTube app, you can download and install it to watch videos.

User: Is there a preferred method for "Join another zoom meeting with id 13572468"?

CogAgent: Plan: 1. Identify and select the 'Join' option for an existing or new meeting ID.

2. Enter the specific Meeting ID when prompted to join a particular Zoom meeting.

3. Confirm any additional required details such as passcodes if necessary, and then join the meeting successfully.

Next Action: Tap on the 'Join' button located in the middle of the screen, which corresponds to the coordinates [x,y]=[0.456,0.18]. This action is intended to initiate the process of joining a Zoom meeting.

Grounded Operation: tap [[456,180]]

User: In how many minutes will the next meeting start? Please think step by step.

CogAgent: The next meeting will start in 23 minutes. This is determined by subtracting the current time (22:35) from the scheduled start time of the meeting (23:00).

<!-- Media -->

<table><tr><td><img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_13.jpg?x=170&y=1905&w=32&h=22&r=0"/></td><td><img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_13.jpg?x=247&y=1903&w=28&h=26&r=0"/></td><td><img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_13.jpg?x=323&y=1904&w=28&h=23&r=0"/></td><td><img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_13.jpg?x=399&y=1903&w=29&h=27&r=0"/></td><td>...</td></tr><tr><td>Meetings</td><td>Team Chat</td><td>Mail</td><td>Calendar</td><td>More</td></tr></table>

<!-- Media -->

Figure 16. More generated samples of CogAgent.


<!-- Media -->

<!-- figureText: ✘<br>»<br>Arxiv-sanity bot @arxivsanitybot Q Automated<br>tinyurl.com/ytk7wbdh A new open-source visual language model called CogVLM has been introduced, which allows for deep fusion of vision language features and achieves state-of-the-art performance on various benchmarks.<br>NoCaps<br>VQAv2<br>COCO-caption<br>48.75<br>VizWiz-VQA<br>Flickr30K<br>123.2<br>42.5<br>72.8<br>90.0<br>9:27 PM - Nov 7, 2023 - 161 Views<br>TextVQA<br>93.85<br>RefCOCO<br>O<br>a<br>0<br>贝<br>↑<br>46.75<br>89.95<br>87.25<br>89.5<br>Post your reply<br>Reply<br>1.368045<br>1.75<br>91.15<br>OKVQA<br>RefCOCO+<br>Discover more<br>58.11<br>92.72<br>0.29<br>BLIP2<br>Sourced from across X<br>InstructBLIP<br>Arxiv-sanity @arxivsanit . Nov 7 .. - Automated tinyurl.com/ygclvkvk A framework for classifying the capabilities and behavior of Artificial General Intelligence models and their precursors, with the hope of providing a common language to compare models, assess risks, and measure progress along the path to AGI.<br>64.87<br>Shikra-13B<br>ScienceQA-IMG<br>RefCOCOg<br>Qwen-VL<br>PaL1-17B<br>PaLI-X-55B<br>GQA<br>Visual7W<br>PaLM-E-84B<br>CogVLM-17B<br>Q<br>12. 1<br>0<br>111 161<br>↑ -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_14.jpg?x=284&y=222&w=1186&h=880&r=0"/>

User: What's the performance on CogVLM-17B on Visual7W?

CogAgent: The performance on CogVLM-17B on Visual7W is 90.75.

User: What's Arxiv-sanity's comment?

CogAgent: Arxiv-sanity's comment is: 'A new open-source visual language model called CogVLM has been introduced, which allows for deep fusion of vision language features and achieves state-of-the-art performance on various benchmarks.'

<!-- figureText: CAPRICCIOSA 8,50-6<br>DIAVOLA<br>4 STAGION1'8 SO<br>MARIWARA<br>Saporita<br>soc<br>WRSTEL<br>APRESE<br>50€<br>prosciute<br>F. 90€<br>Gorgonzollar 70€<br>VEGETARIANA 8,50€<br>FUNGHI<br>6 10€ -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_14.jpg?x=286&y=1353&w=624&h=474&r=0"/>

<!-- figureText: Lamp doesn't work<br>Lamp plugged in?<br>Plug in lamp<br>Bulb burned out?<br>Yes<br>Replace bulb<br>No<br>立<br>Repair lamp -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_14.jpg?x=1031&y=1342&w=373&h=479&r=0"/>

User: What is the solution if the lamp is plugged in and the bulb has burned out? CogAgent: If the lamp is plugged in and the bulb has burned out, the solution is to 'Replace bulb'.

User: I am getting one funghi pizza and one prosciutto pizza. How much should I pay in total?

CogAgent: The funghi pizza costs 6,90\\u20ac and the prosciutto pizza costs 7,90\\u20ac. Therefore, the total cost for both pizzas would be 14,80\\u20ac.

Figure 17. More generated samples of CogAgent.


<!-- figureText: Product History<br>The SleekSneaker was first released in 2001. It quickly becomes a popular choice for athletes people of all ages and activity levels<br>2001<br>The SleekSneaker is first released<br>The second generation of SleekSneakers was released in 2008. Building on the original's success, this new version features updated styling and a more breathable mesh upper. It remains a popular choice for athletes and casual wearers alike.<br>2008<br>The second generation of SleekSneakers is released<br>The third convention of Slack Sneakers was released in 2018. This latest version features a more modern design and a new, dumble synthetic upper. It is quickly embraced by athletes and everyday wearers alike for its comfort, style, and dumbility<br>2018<br>The third generation of<br>The fourth generation of SleekSneakers was released in 2022. This version features an<br>2022<br>The fourth generation of SleekSneakers is released<br>4<br>The fifth reneration of SleekSneakers will be released in 2024. This version features a new, and casual wearers. Thanks to its updated design and improved durability, the SleekSneaker will become a go-to choice for active people of all ages.<br>2024<br>The fifth generation of SleekSneakers is released -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_15.jpg?x=291&y=330&w=1174&h=666&r=0"/>

User: Describe this image.

CogAgent: The image is a timeline showcasing the product history of 'SleekSneakers'. It starts from 2001, where 'The SleekSneaker' was first released, and ends in 2024, where the fifth generation of SleekSneakers was released. The timeline highlights key milestones, such as the introduction of the second generation of SleekSneakers in 2008, the fourth generation in 2022, and the fifth generation in 2024. Each milestone is accompanied by a brief description.

User: Read the content in 2022.

<!-- Media -->

CogAgent: In 2022, the fourth generation of SleekSneakers was released. This version features an updated design with a new, more breathable mesh upper. It is suitable for all activities and remains popular for athletes and casual wearers.

<!-- Media -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_15.jpg?x=287&y=1362&w=656&h=375&r=0"/>

User: In which country was this photo taken? CogAgent: The photo was taken in Australia, as the Australian Open is one of the major tennis tournaments held there.

User: Who is the person on the left?

CogAgent: The person on the left is Keira

Knightley.

<!-- Media -->

Figure 18. More generated samples of CogAgent.


<!-- Media -->

<table><tr><td>method</td><td>top-1 err.</td><td>top-5 err.</td></tr><tr><td>VGG [41] (ILSVRC'14)</td><td>-</td><td>8.43†</td></tr><tr><td>GoogLeNet [44] (ILSVRC'14)</td><td>-</td><td>7.89</td></tr><tr><td>VGG [41] (v5)</td><td>24.4</td><td>7.1</td></tr><tr><td>PReLU-net [13]</td><td>21.59</td><td>5.71</td></tr><tr><td>BN-inception [16]</td><td>21.99</td><td>5.81</td></tr><tr><td>ResNet-34 B</td><td>21.84</td><td>5.71</td></tr><tr><td>ResNet-34 C</td><td>21.53</td><td>5.60</td></tr><tr><td>ResNet-50</td><td>20.74</td><td>5.25</td></tr><tr><td>ResNet-101</td><td>19.87</td><td>4.60</td></tr><tr><td>ResNet-152</td><td>19.38</td><td>4.49</td></tr></table>

Table 4. Error rates (%) of single-model results on the ImageNet validation set (except ${}^{ \dagger  }$ reported on the test set).

---

	User: Convert it to the markdown format.

	CogAgent: | method | top-1 err. | top-5 err. |

<!-- Media -->

<!-- figureText: | -------- | ------------ | ------------ |<br>| VGG [41] (ILSVRC'14)<br>18.43<br>| GoogLeNet [44] (ILSVRC'14) | - | 7.89 |<br>| VGG [41] (v5) | 24.4 | 7.1<br>| PReLU-net [13] | 21.59 | 5.71<br>| BN-inception [16] | 21.99 | 5.81 |<br>| ResNet-34 B | 21.84 | 5.71 |<br>| ResNet-34 C | 21.53 | 5.60<br>| ResNet-50 | 20.74 | 5.25 |<br>| ResNet-101 | 19.87 | 4.60 |<br>[ResNet-152 | 19.38 | 4.49 | -->

	<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_16.jpg?x=181&y=829&w=519&h=321&r=0"/>

<!-- Media -->

---

Dataset 1

<table><tr><td>Department</td><td>Name</td><td>Salary</td></tr><tr><td>Administration</td><td>Emily</td><td>\$2,000</td></tr><tr><td>Finance</td><td>Nick</td><td>\$1,800</td></tr><tr><td>Marketing</td><td>John</td><td>\$2,600</td></tr><tr><td>Marketing</td><td>Gloria</td><td>\$1,800</td></tr><tr><td>Administration</td><td>Jane</td><td>\$2,500</td></tr><tr><td>Finance</td><td>Max</td><td>\$2,000</td></tr><tr><td>Administration</td><td>Tyler</td><td>\$3,000</td></tr><tr><td>Finance</td><td>Wilkins</td><td>\$2,500</td></tr><tr><td>Marketing</td><td>Miles</td><td>\$2,800</td></tr><tr><td>Administration</td><td>Sabine</td><td>\$1,900</td></tr></table>

User: Based on the table, how much is Wilkins earning?

CogAgent: Wilkins is earning \$2,500.

US Nationwide:

<!-- figureText: Median House Price - Median Gross Rent per Month<br>Median Household Income<br>Growth rate:<br>150%<br>100%<br>50%<br>0<br>1960<br>1970<br>1980<br>1990<br>2000<br>2008<br>2010<br>2017 -->

<img src="https://cdn.noedgeai.com/bo_d44nd54601uc738n6peg_16.jpg?x=895&y=326&w=715&h=514&r=0"/>

User: Which year has the highest growth rate of median house price?

CogAgent: The year 2008 has the highest growth rate of median house price, which is 150%.

<!-- Media -->

<!-- Meanless: Figure 19. More gener<br>Figure 19. More generated samples of CogAgent. -->