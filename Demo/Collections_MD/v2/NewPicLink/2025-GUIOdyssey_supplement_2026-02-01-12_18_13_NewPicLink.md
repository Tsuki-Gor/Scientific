# GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices
# GUIOdyssey：用于移动设备跨应用 GUI 导航的综合数据集


Supplementary Material
附加材料


## 7. Ethical Discussion
## 7. 伦理讨论


Privacy. We use temporary accounts and virtual user-names to register various apps and ensure no personal information is entered. The dataset contains no authentic personal information.
隐私。我们使用临时账户和虚拟用户名注册各种应用，确保不输入个人信息。数据集不包含任何真实个人信息。


Ethical Consent in Data Collection. A formal consent process is implemented, wherein participants explicitly agree to the inclusion of their human-annotated data in the dataset. All data are collected with informed consent and in full compliance with ethical guidelines.
数据收集中的伦理同意。实现了正式的同意流程，参与者明确同意将其人工标注数据纳入数据集。所有数据在知情同意下收集，完全符合伦理准则。


Security Concerns. The development of intelligent agents trained on datasets like this offers significant potential for automating tasks and enhancing accessibility. However, it also raises important ethical and security concerns. Sensitive operations, such as financial transactions or privacy management, pose vulnerabilities without robust safeguards. Additionally, malicious actors could exploit these agents to bypass security protocols or manipulate applications for unethical purposes. To mitigate these risks, it is crucial to implement secure model designs, privacy-preserving techniques, and establish clear ethical guidelines. Addressing these challenges will help ensure the responsible deployment of such technology while maximizing its societal benefits.
安全性关注。以此类数据集训练的智能代理在自动化任务和提升可访问性方面具有显著潜力。然而，也带来重要的伦理与安全问题。敏感操作如金融交易或隐私管理在缺乏强健保护时存在脆弱性。此外，恶意行为者可能利用这些代理绕过安全协议或以不道德目的操纵应用。为降低风险，关键在于实现安全的模型设计、隐私保护技术，并建立明确的伦理准则。解决这些挑战将有助于负责任地部署此类技术，同时最大化其社会收益。


## 8. Details of GUIOdyssey
## 8. GUIOdyssey 的详细信息


### 8.1. Description of Task Categories
### 8.1. 任务类别描述


The specific details of the six task categories are as follows:
六个任务类别的具体细节如下：


General Tool. This category encompasses tasks that involve navigating through system-wide operations such as managing system settings or notifications for apps. An instruction example of a general tool task is "Adjust the notification settings for the YouTube app on your phone using Settings, then proceed to open YouTube".
通用工具。此类别涵盖涉及通过系统范围操作进行导航的任务，如管理应用的系统设置或通知。一个通用工具任务的示例指令是“在手机的设置中调整 YouTube 应用的通知设置，然后打开 YouTube”。


Information Management. Information management tasks involve searching for information and recording it for future use. This might include looking up information on search engines, reading articles on news apps, checking facts on educational or reference apps, and then saving or organizing this information in note-taking apps.
信息管理。信息管理任务涉及为将来使用而搜索信息并记录信息。这可能包括在搜索引擎上查找信息、在新闻应用中阅读文章、在教育或参考应用中核对事实，然后将这些信息保存或在笔记应用中进行整理。


Web Shopping. Shopping tasks encompass a range of activities related to purchasing products online. Users may start by searching for a product on one app, comparing prices on different e-commerce platforms, checking reviews and ratings on review apps or websites, and finally making a purchase.
网络购物。购物任务涵盖在线购买产品的各种活动。用户可能先在一个应用中搜索产品，在不同的电商平台上比较价格，在评价应用或网站上查看评价与评分，最后完成购买。


Media Entertainment. Media entertainment tasks are about activities involving video and music streaming apps. Users may browse for new content on video platforms like YouTube or Netflix, stream music on services like Spotify or Apple Music, and switch between different media apps to manage playlists or download content.
媒体娱乐。媒体娱乐任务涉及视频和音乐流媒体应用的活动。用户可能在 YouTube、Netflix 等视频平台上浏览新内容，在 Spotify、Apple Music 等服务上进行音乐串流，并在不同的媒体应用之间切换以管理播放列表或下载内容。


Social Sharing. This task involves activities where users share content across different social media platforms. This could include taking photos or videos with the camera app, editing them using a photo or video editing app, and then sharing them on multiple social media platforms like Insta-gram, Facebook, Twitter, or TikTok.
社交分享。此任务涉及用户在不同社交媒体平台上分享内容的活动。可能包括使用相机应用拍摄照片或视频，使用照片或视频编辑应用进行编辑，然后在 Instagram、Facebook、Twitter、TikTok 等多个社交平台上分享。


Multi-Apps. Multiple-app tasks involve more complex operations that require three or more apps to complete. For example, cooking food with an online recipe might involve finding the recipe of the food, recording the recipe to a note-taking app, and buying the ingredients online(Fig. 1).
多应用。多应用任务涉及需要三个以上应用才能完成的更复杂的操作。例如，使用在线食谱烹饪食物可能包括查找食谱、将食谱记录到笔记应用、并在线购买材料（图 1）。


### 8.2. Action Set
### 8.2. 动作集


Our recording system utilizes Android Studio to simulate GUI navigation and virtualize various devices. We use the Android Debug Bridge (ADB) to retrieve device information and status, such as the coordinates of click events, and to monitor a wide range of functional keys. The details of the action set in our Android emulator are presented in Table 5.
我们的记录系统使用 Android Studio 来模拟 GUI 导航并虚拟化多种设备。我们利用 Android Debug Bridge (ADB) 获取设备信息和状态，例如点击事件的坐标，并监控广泛的功能键。Android 模拟器中的操作集细节在表 5 中展示。


### 8.3. Fine-grained Episode Annotation Generation
### 8.3. 细粒度 Episode 注释生成


Fine-grained episode annotations consist of two components: low-level instructions and semantic annotations. Examples of the fine-grained annotations can be found in Fig. 7.
细粒度的 episode 注释由两部分组成：低级指令和语义注释。细粒度注释的示例可见图 7。


Low-Level Instruction. For each step within an episode, we provide GPT-40 with the high-level instruction corresponding to the episode, along with the action and screenshot associated with the current step. Additionally, for actions such as CLICK and LONG PRESS, we supply an additional image featuring a bounding box to indicate the click coordinates. All images are configured with the fidelity parameter set to 'high'. The prompt utilized is provided in Fig. 11.
低级指令。对于每个 episode 的每一步，我们向 GPT-40 提供与该 episode 相对应的高级指令，以及与当前步骤相关的操作和截图。此外，对于 CLICK 和 LONG PRESS 等操作，我们提供一个额外的图像，包含边界框以标示点击坐标。所有图像的保真度参数设置为 'high'。所用的提示如图 11 所示。


Semantic Annotation. We use GPT-40 to generate semantic annotations in an alternating and iterative manner, following the sequential order of steps within each episode. Specifically, the process begins by providing the current episode's high-level instruction along with the actions and decision rationale from previous steps, prompting GPT-4o to generate the contextual information for the current step. Subsequently, using the generated contextual information, the high-level instruction, the screenshot image, and the action corresponding to the current step, GPT-40 is prompted step-by-step to generate the screen description and decision rationale for the current step. This iterative process continues until all semantic annotations for each step within the episode are completed in sequence. Similarly, for actions such as CLICK and LONG PRESS, we supply an additional image with a bounding box indicating the click coordinates. All images are configured with the fidelity parameter set to 'high' to ensure precision. The prompts used for generating these annotations are provided in Fig. 12 and Fig. 13.
语义注释。我们以交替迭代的方式使用 GPT-40 生成语义注释，按照每个 episode 内步骤的顺序进行。具体地，过程先提供当前 episode 的高级指令，以及前面步骤的操作和决策理由，提示 GPT-4o 生成当前步骤的上下文信息。随后，利用生成的上下文信息、高级指令、当前步骤的截图图像以及对应的操作，逐步提示 GPT-40 生成当前步骤的屏幕描述和决策理由。这一迭代过程持续进行，直到该 episode 内每一步的语义注释全部按序完成。类似地，对于 CLICK 和 LONG PRESS 等操作，我们提供一个额外的带有点击坐标边界框的图像。所有图像的保真度参数设置为 'high' 以确保精确性。用于生成这些注释的提示如图 12 和图 13 所示。


Table 5. The argument and functionality of different actions in GUIOdyssey. 'pos1' and 'pos2' denote the position $\left( {x,y}\right)$ .
表 5. GUIOdyssey 中不同操作的参数与功能。'pos1' 与 'pos2' 表示位置 $\left( {x,y}\right)$ 。


<table><tr><td>Action</td><td>Argument</td><td>Functionality</td></tr><tr><td>CLICK</td><td>[pos1]</td><td>click the on-screen position</td></tr><tr><td>LONG PRESS</td><td>[pos1]</td><td>press the screen for a long time to copy texts or download images</td></tr><tr><td>SCROLL</td><td>[pos1, pos2]</td><td>scroll the screen from position 1 to position 2</td></tr><tr><td>TYPE</td><td>text</td><td>type text with keyboard</td></tr><tr><td>COMPLETE</td><td>-</td><td>the sign that the instruction has been completed</td></tr><tr><td>IMPOSSIBLE</td><td>-</td><td>the sign that the instruction cannot be completed</td></tr><tr><td>HOME</td><td>-</td><td>go to the home screen</td></tr><tr><td>BACK</td><td>-</td><td>go to the previous screen</td></tr><tr><td>RECENT</td><td>-</td><td>go to the previous App</td></tr></table>
<table><tbody><tr><td>动作</td><td>参数</td><td>功能</td></tr><tr><td>点击</td><td>[pos1]</td><td>点击屏幕上的位置</td></tr><tr><td>长按</td><td>[pos1]</td><td>长按屏幕以复制文本或下载图片</td></tr><tr><td>滚动</td><td>[pos1, pos2]</td><td>从位置1滚动到位置2</td></tr><tr><td>输入</td><td>文本</td><td>用键盘输入文本</td></tr><tr><td>完成</td><td>-</td><td>指示任务已完成的标志</td></tr><tr><td>不可能</td><td>-</td><td>指示任务无法完成的标志</td></tr><tr><td>主页</td><td>-</td><td>返回主屏幕</td></tr><tr><td>返回</td><td>-</td><td>返回到上一屏幕</td></tr><tr><td>最近</td><td>-</td><td>转到上一个应用</td></tr></tbody></table>


### 8.4. Examples
### 8.4. 例子


An example of episodes in our GUIOdyssey is shown in Fig. 6, while examples of semantic annotations can be found in Fig. 7. An example of an annotation for a task that could not be successfully completed and ends with the IMPOSSIBLE action can be found in Fig. 8 and Fig. 9.
在我们的 GUIOdyssey 中的一个剧集示例显示在图6，而语义注解的示例可在图7中找到。一个关于无法成功完成并以 IMPOSSIBLE 行为结束的任务注解的示例可在图8和图9中找到。


As mentioned in Sec. 5.1, we use SAM2 [37] to assist in evaluating whether the model's output actions are correct. Fig. 10 provides examples of bounding boxes for clicked elements obtained through SAM2 segmentation.
如第5.1节所述，我们使用 SAM2 [37] 来辅助评估模型的输出动作是否正确。图10 提供通过 SAM2 分割获得的被点击元素的边界框示例。


### 8.5. Data Format
### 8.5. 数据格式


Each field of annotation is as follows.
注解的每个字段如下。


episode_id: the unique identifier of this episode.
episode_id：本剧集的唯一标识符。


device_info: the detailed information of the virtual device from which the episode was collected, including the device model, screen resolution, and other device-related details.
device_info：收集该剧集所使用的虚拟设备的详细信息，包括设备型号、屏幕分辨率和其他设备相关细节。


task_info: the detailed information of the task from which the episode was collected, including the task category, the app used, the high-level instruction, and other task-related details. step_length: the total number of steps in this episode.
task_info：剧集所对应任务的详细信息，包括任务类别、所用应用、高层指令等任务相关细节。step_length：本剧集中的总步数。


steps: a list of steps in this episode. Each step in the list includes the file path of the screenshot, executed action and its corresponding parameters (e.g., the coordinates for a click action), the low-level instruction, the semantic annotation, the bounding box obtained from SAM2 segmentation, and additional recorded information such as the overall scroll trajectory for scroll actions and annotator notes.
steps：本剧集中的步骤列表。列表中的每个步骤包括截图的文件路径、执行的动作及其对应参数（如点击动作的坐标）、低级指令、语义注解、通过 SAM2 分割得到的边界框，以及如滚动动作的整体滚动轨迹等额外记录信息和标注者注记。


## 9. Experiment Details
## 9. 实验细节


### 9.1. Detailed description of four different setups.
### 9.1. 四种不同设置的详细描述


The following details the four different setups in GUIOdyssey.
下文详细说明 GUIOdyssey 的四种不同设置。


i) Train-Random & Test-Random. We randomly partitioned all the episodes in the dataset into training and testing sets using a ratio of ${80}\%$ to ${20}\%$ as the standard approach to divide the dataset. It can assess the in-domain performance of OdysseyAgent.
i) Train-Random &amp; Test-Random。我们将数据集中所有剧集随机划分为训练集和测试集，按 ${80}\%$ 与 ${20}\%$ 的比例作为划分数据集的标准方法。它可以评估 OdysseyAgent 的领域内表现。


ii) Train-Task & Test-Task. In this setup, We proportionally sampled meta-tasks from six categories, maintaining approximately a $6 : 1$ ratio for the training and test sets. The tasks in the test set differ significantly from those in the training set. This partitioning method allows for a robust assessment of an agent's generalization capabilities across diverse tasks.
ii) Train-Task &amp; Test-Task。在此设置中，我们从六个类别中成比例抽样元任务，训练集与测试集保持约 $6 : 1$ 的比例。测试集中的任务与训练集中的任务显著不同。这种划分方法有助于对智能体在多样任务中的泛化能力进行稳健评估。


iii) Train-Device & Test-Device. To evaluate an agent's generalizability across different and unseen devices, we selected episodes annotated on the Tablet, which differs significantly from other devices, as the test set. We obtained 1,381 episodes as the test set and 6,953 episodes as the training set.
iii) Train-Device &amp; Test-Device。为评估智能体在不同且未见设备上的泛化能力，我们选择在平板电脑上标注的剧集作为测试集，这些设备与其他设备差异显著。我们获得了1,381个剧集作为测试集，6,953个剧集作为训练集。


iv) Train-App & Test-App. This split is aimed at evaluating the agent's performance on unseen Apps and App combinations. First, we calculated the frequency of app usage in the dataset and categorized the apps into 25 classes (e.g., Video, Music) based on their characteristics. Then, we selected a few apps with the lowest occurrence from each class to form the test app set. Subsequently, we partitioned the episodes that utilized the app in the test app set into the Test-App set, maintaining an approximately 85% to 15% ratio between the training set and the test set.
iv) Train-App &amp; Test-App。此划分旨在评估智能体在未见应用及应用组合上的性能。首先，我们计算数据集中应用的使用频率，并基于其特征将应用分类为25个类别（如视频、音乐）。然后，从每个类别中选择出现频率最低的若干应用组成测试应用集。随后，我们将使用测试应用集中的应用所涉及的剧集划分到 Test-App 集，训练集与测试集的比例约为 85% 对 15%。


Table 6. The impact of different semantic annotations on OdysseyAgent across four different splits. We use high-level instructions for both training and evaluation. Performance is assessed using AMS and SR as metrics. SD, CI, and DR denote screen description, contextual information, and decision rationale, respectively.
表6. 不同语义注释对 OdysseyAgent 在四个不同拆分上的影响。训练与评估均使用高层指令。性能以 AMS 和 SR 为指标进行评估。SD、CI 与 DR 分别表示屏幕描述、上下文信息与决策理由。


<table><tr><td rowspan="2"></td><td colspan="3">Semantic Annotation</td><td colspan="2">Test-Random</td><td colspan="2">Test-Task</td><td colspan="2">Test-Device</td><td colspan="2">Test-App</td><td colspan="2">Overall</td></tr><tr><td>SD</td><td>CI</td><td>DR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td></tr><tr><td>(1)</td><td>✘</td><td>✘</td><td>✘</td><td>75.79</td><td>9.38</td><td>54.36</td><td>0.09</td><td>61.20</td><td>1.88</td><td>63.03</td><td>7.70</td><td>63.60</td><td>4.76</td></tr><tr><td>(2)</td><td>✓</td><td>✘</td><td>✘</td><td>75.18</td><td>8.94</td><td>54.06</td><td>0.00</td><td>64.41</td><td>2.03</td><td>64.91</td><td>8.47</td><td>64.64</td><td>4.86</td></tr><tr><td>(3)</td><td>✘</td><td>✓</td><td>✘</td><td>75.42</td><td>10.04</td><td>55.71</td><td>0.00</td><td>62.52</td><td>3.19</td><td>64.24</td><td>5.30</td><td>64.47</td><td>4.63</td></tr><tr><td>(4)</td><td>✘</td><td>✘</td><td>✓</td><td>77.71</td><td>11.44</td><td>55.60</td><td>0.26</td><td>65.88</td><td>4.63</td><td>65.74</td><td>7.96</td><td>66.23</td><td>6.07</td></tr><tr><td>(5)</td><td>✘</td><td>✓</td><td>✓</td><td>77.23</td><td>11.16</td><td>56.93</td><td>0.18</td><td>63.87</td><td>2.24</td><td>66.32</td><td>7.87</td><td>66.09</td><td>5.36</td></tr><tr><td>(6)</td><td>✓</td><td>✘</td><td>✓</td><td>77.24</td><td>10.88</td><td>57.15</td><td>0.00</td><td>63.55</td><td>2.17</td><td>67.04</td><td>9.67</td><td>66.24</td><td>5.68</td></tr><tr><td>(7)</td><td>✓</td><td>✓</td><td>✘</td><td>76.58</td><td>10.14</td><td>57.13</td><td>0.26</td><td>64.48</td><td>3.91</td><td>66.27</td><td>7.96</td><td>66.11</td><td>5.57</td></tr><tr><td>(8)</td><td>✓</td><td>✓</td><td>✓</td><td>78.24</td><td>11.62</td><td>56.19</td><td>0.26</td><td>66.63</td><td>5.07</td><td>65.89</td><td>8.81</td><td>66.74</td><td>6.44</td></tr></table>
<table><tbody><tr><td rowspan="2"></td><td colspan="3">语义标注</td><td colspan="2">随机测试</td><td colspan="2">测试任务</td><td colspan="2">测试设备</td><td colspan="2">测试应用</td><td colspan="2">总体</td></tr><tr><td>SD</td><td>CI</td><td>DR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td></tr><tr><td>(1)</td><td>✘</td><td>✘</td><td>✘</td><td>75.79</td><td>9.38</td><td>54.36</td><td>0.09</td><td>61.20</td><td>1.88</td><td>63.03</td><td>7.70</td><td>63.60</td><td>4.76</td></tr><tr><td>(2)</td><td>✓</td><td>✘</td><td>✘</td><td>75.18</td><td>8.94</td><td>54.06</td><td>0.00</td><td>64.41</td><td>2.03</td><td>64.91</td><td>8.47</td><td>64.64</td><td>4.86</td></tr><tr><td>(3)</td><td>✘</td><td>✓</td><td>✘</td><td>75.42</td><td>10.04</td><td>55.71</td><td>0.00</td><td>62.52</td><td>3.19</td><td>64.24</td><td>5.30</td><td>64.47</td><td>4.63</td></tr><tr><td>(4)</td><td>✘</td><td>✘</td><td>✓</td><td>77.71</td><td>11.44</td><td>55.60</td><td>0.26</td><td>65.88</td><td>4.63</td><td>65.74</td><td>7.96</td><td>66.23</td><td>6.07</td></tr><tr><td>(5)</td><td>✘</td><td>✓</td><td>✓</td><td>77.23</td><td>11.16</td><td>56.93</td><td>0.18</td><td>63.87</td><td>2.24</td><td>66.32</td><td>7.87</td><td>66.09</td><td>5.36</td></tr><tr><td>(6)</td><td>✓</td><td>✘</td><td>✓</td><td>77.24</td><td>10.88</td><td>57.15</td><td>0.00</td><td>63.55</td><td>2.17</td><td>67.04</td><td>9.67</td><td>66.24</td><td>5.68</td></tr><tr><td>(7)</td><td>✓</td><td>✓</td><td>✘</td><td>76.58</td><td>10.14</td><td>57.13</td><td>0.26</td><td>64.48</td><td>3.91</td><td>66.27</td><td>7.96</td><td>66.11</td><td>5.57</td></tr><tr><td>(8)</td><td>✓</td><td>✓</td><td>✓</td><td>78.24</td><td>11.62</td><td>56.19</td><td>0.26</td><td>66.63</td><td>5.07</td><td>65.89</td><td>8.81</td><td>66.74</td><td>6.44</td></tr></tbody></table>


### 9.2. Training Details.
### 9.2. 训练细节。


To train OdysseyAgent, we employ the AdamW optimizer with a learning rate of ${2e} - 5$ and utilize a cosine learning rate schedule. We set ${\beta }_{1}$ and ${\beta }_{2}$ to 0.9 and 0.95,respectively, and use a weight decay of 0.1 . Additionally, we utilize a global batch size of 128 and implement DeepSpeed ZERO2-style data parallelism. During training, OdysseyA-gent treats each action step as an individual training sample. The input consists of the task instruction, the current screenshot, and the previous 4 actions and screenshots (i.e., $\delta  = 4$ ),while the output corresponds to the action for the current step. By default, OdysseyAgent is trained separately on Train-Random/Task/Device/App for one epoch, excluding the semantic annotation component. When training includes semantic annotations, these annotations are converted into single-turn QA pairs, which serve as additional training samples (i.e., semantic annotations are introduced only during training-time). Any training configuration that incorporates semantic annotations is explicitly noted. The entire training process requires approximately 32 A100 hours to complete.
为了训练 OdysseyAgent，我们使用 AdamW 优化器，学习率为 ${2e} - 5$，并采用余弦学习率调度。我们将 ${\beta }_{1}$ 和 ${\beta }_{2}$ 分别设为 0.9 和 0.95，权重衰减为 0.1。此外，我们使用全局批大小 128，并实现 DeepSpeed ZERO2 风格的数据并行。在训练过程中，OdysseyA-gent 将每个动作步骤视为一个独立的训练样本。输入包括任务指令、当前截图以及前 4 个动作和截图（即 $\delta  = 4$），输出对应当前步骤的动作。默认情况下，OdysseyAgent 在 Train-Random/Task/Device/App 上单独训练一个时期，不包含语义注释组件。当训练包含语义注释时，这些注释被转换为单轮问答对，作为额外的训练样本（即语义注释仅在训练时引入）。任何包含语义注释的训练配置都将被明确标注。整个训练过程大约需要 32 个 A100 小时完成。


### 9.3. Prompt for Evaluation.
### 9.3. 评测提示。


We utilize the prompt shown in Fig. 14 to evaluate the performance of GPT-4V, GPT-4o, Claude3.5-sonnet, and InternVL2-Pro. For SphAgent and CogAgent, we tested them following their officially recommended methods [9, 23].
我们使用图 14 中给出的提示来评估 GPT-4V、GPT-4o、Claude3.5-sonnet 与 InternVL2-Pro 的性能。对于 SphAgent 和 CogAgent，我们按其官方推荐方法进行测试 [9, 23]。


## 10. More Experiments
## 10. 更多实验


### 10.1. History Resampler vs. Multi-Image Training.
### 10.1. 历史重采样与多图像训练。


We evaluate different approaches for processing historical screenshot images. Qwen-VL supports multi-image input by interleaving image and text tokens, but this leads to a high token overhead (e.g., 1024 tokens for four historical steps). Our history resampler compresses this to 256 tokens, greatly improving efficiency. As shown in Table 7, both approaches achieve comparable performance, but the history resampler significantly enhances training and inference efficiency.
我们评估处理历史截图图像的不同方法。Qwen-VL 通过交错图像和文本标记来支持多图像输入，但这会导致较高的标记开销（如四个历史步骤需 1024 个标记）。我们的历史重采样将其压缩到 256 个标记，大幅提升效率。如表 7 所示，两种方法的性能相近，但历史重采样显著提升了训练和推理效率。


Table 7. The average AMS for HL and LL instructions across 4 splits, along with the number of historical screenshot tokens, inference metrics (Time to First Token (TTFT) and Tokens per Second (TPS)), and training GPU hours.
Table 7. HL 与 LL 指令在 4 个分割上的平均 AMS，以及历史截图标记数量、推理指标（首个标记耗时 TTFT 和每秒标记数 TPS）和训练 GPU 小时数。


<table><tr><td>strategy</td><td>HL</td><td>LL</td><td>Token Count</td><td>TTFT $\downarrow$</td><td>TPS $\uparrow$</td><td>GPU Hours</td></tr><tr><td>history resampler</td><td>63.60</td><td>82.44</td><td>256</td><td>0.71</td><td>20.27</td><td>32</td></tr><tr><td>multi-image</td><td>65.04</td><td>82.34</td><td>1024</td><td>0.98</td><td>17.05</td><td>48</td></tr></table>
<table><tbody><tr><td>策略</td><td>HL</td><td>LL</td><td>令牌计数</td><td>TTFT $\downarrow$</td><td>TPS $\uparrow$</td><td>GPU 小时</td></tr><tr><td>历史重采样器</td><td>63.60</td><td>82.44</td><td>256</td><td>0.71</td><td>20.27</td><td>32</td></tr><tr><td>多图像</td><td>65.04</td><td>82.34</td><td>1024</td><td>0.98</td><td>17.05</td><td>48</td></tr></tbody></table>


#### 10.2.The effect of different semantic annotations.
#### 10.2. 不同语义注释的影响。


We assess the impact of different semantic annotations in GUIOdyssey (i.e., screen description, contextual information and decision rationale) on model performance in both in-domain and out-of-domain settings. The results are presented in Table 6. A comparison of experiments (1)-(4) shows that all three components contribute positively, but engaging in detailed reasoning before making decisions is more important than understanding current screen information or summarizing historical processes in cross-app tasks. Experiments (5)-(8) further indicate that using two or more types of semantic annotations generally outperforms using a single annotation type. Specifically, using all semantic annotations yields the best results and improves AMS by 3.14 and SR by ${35}\%$ compared to training without any semantic annotations. These findings suggest that teaching the model to understand the reasoning behind each action-similar to how humans observe, understand, review completed steps, and reason thoroughly before deciding—can be beneficial for improving performance in both in-domain and out-of-domain cross-app tasks.
我们评估 GUIOdyssey 中不同语义注释（即屏幕描述、上下文信息和决策依据）对模型在同域和跨域任务中的性能影响。结果见表6。对实验（1）-（4）的比较显示，三种组成部分均有正向贡献，但在跨应用任务中，在做出决定之前进行详细推理比理解当前屏幕信息或总结历史过程更为重要。实验（5）-（8）进一步表明，使用两种以上类型的语义注释通常优于使用单一注释类型。具体而言，使用全部语义注释可获得最佳结果，与不使用任何语义注释的训练相比，AMS 提高 3.14，SR 提高 ${35}\%$。这些发现表明，教会模型理解每个操作背后的推理——类似于人类观察、理解、回顾已完成步骤并在决定前进行充分推理——有助于提升同域和跨域跨应用任务的性能。


### 10.3. Transferability of instructions at different lev- els of granularity.
### 10.3. 不同粒度水平指令的可迁移性。


As shown in Table 8, models trained on high-level instructions exhibit significantly better transferability across different levels of instruction granularity compared to those trained on low-level instructions. Furthermore, training on both instruction granularities outperforms training on a single granularity, a phenomenon similar to what has been observed in single-app tasks [26].
如表8所示，在较高层次指令上训练的模型在跨不同指令粒度水平的可迁移性显著优于在低层次指令上训练的模型。此外，在两种粒度水平上进行训练的结果优于在单一粒度水平上训练的结果，这一现象与单应用任务中的观察类似[26]。


### 10.4. Transferability across different devices.
### 10.4. 跨不同设备的可迁移性。


We utilize our GUIOdyssey dataset to conduct additional experiments to evaluate the generalization capabilities of OdysseyAgent beyond the initial experimental setup. we test the OdysseyAgent's adaptability by using data from one device as the test set while training on data from the remaining five devices. The results of these experiments are presented in the Table 9, demonstrating the model's performance across different devices. The model exhibits the weakest transferability on tablet devices, which we attribute to the significant differences in user interface layouts between tablets and smartphones. Furthermore, the model's transferability on small phones and foldable devices is also suboptimal. We surmise that the disparity in screen resolution compared to other phone models may contribute to this underperformance.
我们利用 GUIOdyssey 数据集开展额外实验，以评估 OdysseyAgent 在初始实验设置之外的泛化能力。我们通过使用一个设备的数据作为测试集，而在其余五个设备的数据上进行训练，来测试 OdysseyAgent 的适应性。这些实验结果见表9，展示模型在不同设备上的表现。模型在平板设备上的可迁移性最差，我们将其归因于平板与智能手机之间的用户界面布局存在显著差异。此外，模型在小型手机和折叠设备上的可迁移性也不理想。我们推断，与其他手机型号相比，屏幕分辨率的差异可能导致这一表现不佳。


Table 8. The results for OdysseyAgent trained and tested on Train-Random/Test-Random with both high-level and low-level instructions are presented, with AMS as the evaluation metric. HL and LL denote high-level and low-level instructions, respectively.
表8。展示在 Train-Random/Test-Random 条件下，使用高层次和低层次指令训练并测试的 OdysseyAgent 的结果，AMS 作为评估指标。HL 指高层次指令，LL 指低层次指令。


<table><tr><td rowspan="2">Testing Instructions</td><td colspan="3">Training Instructions</td></tr><tr><td>HL</td><td>LL</td><td>HL + LL</td></tr><tr><td>HL</td><td>75.79</td><td>29.39</td><td>78.96</td></tr><tr><td>LL</td><td>71.26</td><td>86.88</td><td>88.84</td></tr></table>
<table><tbody><tr><td rowspan="2">测试说明</td><td colspan="3">训练说明</td></tr><tr><td>HL</td><td>LL</td><td>HL + LL</td></tr><tr><td>HL</td><td>75.79</td><td>29.39</td><td>78.96</td></tr><tr><td>LL</td><td>71.26</td><td>86.88</td><td>88.84</td></tr></tbody></table>


### 10.5. Whether cross-App tasks benefit single-App tasks.
### 10.5. 跨应用任务是否有益于单应用任务。


We further investigate whether cross-app tasks benefit single-app performance by evaluating the impact of different training data compositions under controlled conditions. Specifically, we randomly sample 50k training samples each from GUIOdyssey, AITW, and AndroidControl
我们进一步研究在受控条件下，不同训练数据组成对跨应用任务是否有益于单应用性能。具体而言，我们随机从 GUIOdyssey、AITW 和 AndroidControl 各抽取 50k 条训练样本


Table 9. Performance Evaluation of OdysseyAgent Across Different Devices. Each Device serves as a test set while the remaining five devices are used as training sets.
表 9。OdysseyAgent 在不同设备上的性能评估。每个设备作为测试集，剩余五个设备作为训练集。


<table><tr><td>Evaluation Device</td><td>Resolution</td><td>AMS</td><td>SR</td></tr><tr><td>Pixel 7 Pro</td><td>1,440×3,120</td><td>75.91</td><td>7.44</td></tr><tr><td>Pixel 8 Pro</td><td>1,344×2,992</td><td>74.67</td><td>6.05</td></tr><tr><td>Small Phone</td><td>720×1,280</td><td>71.68</td><td>3.77</td></tr><tr><td>Medium Phone</td><td>1,080×2,400</td><td>73.05</td><td>5.45</td></tr><tr><td>Pixel Fold</td><td>2, ${208} \times  1,{840}$</td><td>67.67</td><td>4.48</td></tr><tr><td>Pixel Tablet</td><td>2,560 × 1,600</td><td>61.20</td><td>1.88</td></tr></table>
<table><tbody><tr><td>评测设备</td><td>分辨率</td><td>AMS</td><td>SR</td></tr><tr><td>Pixel 7 Pro</td><td>1,440×3,120</td><td>75.91</td><td>7.44</td></tr><tr><td>Pixel 8 Pro</td><td>1,344×2,992</td><td>74.67</td><td>6.05</td></tr><tr><td>小型手机</td><td>720×1,280</td><td>71.68</td><td>3.77</td></tr><tr><td>中型手机</td><td>1,080×2,400</td><td>73.05</td><td>5.45</td></tr><tr><td>Pixel Fold</td><td>2, ${208} \times  1,{840}$</td><td>67.67</td><td>4.48</td></tr><tr><td>Pixel Tablet</td><td>2,560 × 1,600</td><td>61.20</td><td>1.88</td></tr></tbody></table>


(denoted as Ody50k, AITW50k, and AC50k, respectively) and evaluate their performance on AndroidControl, which provides both in-domain and out-of-domain scenarios. As shown in Table 10, we find that incorporating cross-app data from GUIOdyssey consistently enhances performance in most single-app scenarios, whereas adding AITW data surprisingly yields limited improvements or even performance degradation. This suggests that the more complex cross-app tasks in GUIOdyssey can benefit single-app tasks.
(表示为 Ody50k、AITW50k 和 AC50k，分别) 并在 AndroidControl 上评估它们的性能，AndroidControl 同时提供域内和域外场景。如表 10 所示，我们发现将 GUIOdyssey 的跨应用数据整合进来在大多数单应用场景中持续提升性能，而添加 AITW 数据令人惊讶地收获有限的改进，甚至可能导致性能下降。这表明 GUIOdyssey 中更复杂的跨应用任务可以使单应用任务受益。


Table 10. Effectiveness of Different Training Data on the Android-Control. The evaluation metrics are the action matching score (AMS).
表 10. 不同训练数据对 Android-Control 的有效性。评估指标为动作匹配分数 (AMS)。


<table><tr><td>Training Data</td><td>IDD</td><td>category_unseen</td><td>app_unseen</td><td>task_unseen</td><td>Overall</td></tr><tr><td>AC50k</td><td>60.43</td><td>54.46</td><td>50.00</td><td>72.10</td><td>59.25</td></tr><tr><td>AC50k + AITW50k</td><td>60.69</td><td>55.26</td><td>45.19</td><td>68.84</td><td>57.50</td></tr><tr><td>AC50k + Ody50k</td><td>61.48</td><td>54.61</td><td>50.96</td><td>72.46</td><td>59.88</td></tr></table>
<table><tbody><tr><td>训练数据</td><td>IDD</td><td>category_unseen</td><td>app_unseen</td><td>task_unseen</td><td>总览</td></tr><tr><td>AC50k</td><td>60.43</td><td>54.46</td><td>50.00</td><td>72.10</td><td>59.25</td></tr><tr><td>AC50k + AITW50k</td><td>60.69</td><td>55.26</td><td>45.19</td><td>68.84</td><td>57.50</td></tr><tr><td>AC50k + Ody50k</td><td>61.48</td><td>54.61</td><td>50.96</td><td>72.46</td><td>59.88</td></tr></tbody></table>


## instruction: Utilize Chrome to research the key property of a Triangle and compile your findings into a concise document using Google Docs.
## 指令：使用 Chrome 研究三角形的关键属性，并将你的发现整理成一份简明文档，使用 Google Docs。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_21_30_d0a127.jpg"/>



Figure 6. An example of episodes in our GUIOdyssey.
图 6。我们 GUIOdyssey 中的一集示例。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_21_30_c4ca5e.jpg"/>



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_21_30_4ae99d.jpg"/>



Figure 8. Example of an annotation for an unsuccessful task, ending with the IMPOSSIBLE action.
图 8。对未成功任务的注释示例，以 IMPOSSIBLE 行动结束。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_21_30_26dec6.jpg"/>



Figure 9. Example of an annotation for an unsuccessful task, ending with the IMPOSSIBLE action.
图 9。对未成功任务的注释示例，以 IMPOSSIBLE 行动结束。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_21_30_e6260b.jpg"/>



Figure 10. Examples of bounding boxes for UI elements segmented by SAM2. The actual click locations are indicated by blue '+' symbols, while the red rectangles outline the bounding boxes obtained from the SAM2.
图 10。由 SAM2 分割的 UI 元素边界框示例。实际点击位置以蓝色“+”符号表示，红色矩形标出由 SAM2 获得的边界框。


---



<img>current_screenshot.png</img>
<img>current_screenshot.png</img>


<img>current_screenshot_w_labels.png</img>
<img>current_screenshot_w_labels.png</img>


Based on the original and marked screenshots of an Android mobile phone, where the marked screenshot is the original screenshot marked with action location, please follow the
基于原始截图与标记的 Android 手机屏幕截图，其中标记截图为在原始截图上标记了动作位置，请遵循下列


instructions below:
指示：


&nbsp;&nbsp;&nbsp;&nbsp;Low-Level Instruction Identification
&nbsp;&nbsp;&nbsp;&nbsp;低级指令识别


- Identify the low-level instruction that the current action represents, such as "Go to the alarm section."
- 识别当前动作所代表的低级指令，例如“前往闹钟区域。”


P.S.:



- When following these instructions, use natural language, avoid mentioning technical details (such as action coordinates) or direct use of action tags (such as "PRESS_HOME").
- 在遵循这些指令时，使用自然语言，避免提及技术细节（如行动坐标）或直接使用行动标签（如“PRESS_HOME”）。


&nbsp;&nbsp;&nbsp;&nbsp;Output Format:
&nbsp;&nbsp;&nbsp;&nbsp;输出格式：


&nbsp;&nbsp;&nbsp;&nbsp;The output should be in JSON format as follows:
&nbsp;&nbsp;&nbsp;&nbsp;输出应以 JSON 格式如下：


\{\{"instruction": "low-level instruction that the current action represents in the desired format"\}\}
{{"instruction": "当前动作所代表的低级指令以所需格式呈现"}}


Please return the result in pure JSON format, without any json tags like ```json ```.
请仅以纯 JSON 格式返回结果，不要包含类似 ```json ``` 的 json 标签。


---



Figure 11. Prompts for generating low-level instruction.
图 11. 生成低级指令的提示。


---



You are completing the task: \{task\} on a mobile phone, and the actions you have performed along with their respective intentions are listed in chronological order as:
你正在手机上完成任务：\{task\}，你所执行的操作及其各自的意图按时间顺序列出如下：


\{intentions\}
\{intentions\}


Please follow the instructions below:
请按照以下指示执行：


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Create a logically connected summary, rather than simply listing each action in detail.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- \t\t- 生成一个逻辑连贯的摘要，而不是逐一详细列出每个动作。


&nbsp;&nbsp;&nbsp;&nbsp;- Avoid using any time-sequence phrases, such as 'after completing,' 'upon finishing,' or similar expressions.
&nbsp;&nbsp;&nbsp;&nbsp;- \t\t- 避免使用任何时间顺序的短语，如“完成后”、“在完成时”或类似表达。


&nbsp;&nbsp;&nbsp;&nbsp;- Use a completed-action tone, describing the progress as if each step has already occurred.
&nbsp;&nbsp;&nbsp;&nbsp;- \t\t- 采用完成行动的语气，描述进展时如同每一步已发生。


&nbsp;&nbsp;&nbsp;&nbsp;- Use an objective tone and describe concisely from an impersonal perspective.
&nbsp;&nbsp;&nbsp;&nbsp;- \t\t- 使用客观语气，从非个人角度简明描述。


&nbsp;&nbsp;&nbsp;&nbsp;- Format the context as follows: "So far, [summary of what has been accomplished]."
&nbsp;&nbsp;&nbsp;&nbsp;- \t\t- 将上下文格式化为如下：“至今为止，[已完成的概要]。”


P.S.:



&nbsp;&nbsp;&nbsp;&nbsp;- When following these instructions, use natural language, avoid mentioning technical details (such as action coordinates) or direct use of action tags (such as "PRESS_HOME").
&nbsp;&nbsp;&nbsp;&nbsp;- \t\t- 在遵循这些指示时，使用自然语言，避免提及技术细节（如动作坐标）或直接使用动作标签（如“PRESS_HOME”）。


Output Format:
输出格式：


The output should be in JSON format as follows:
输出应为如下的 JSON 格式：


\{\{



"context": "a 2-3 sentence summary of task progress up to this point in the desired format"
"context": "到目前为止任务进展的2-3句总结，按所需格式"


\}\}



Please return the result in pure JSON format, without any json tags like ```json ```.
请仅以纯 JSON 格式返回结果，不要包含类似 ```json ``` 的 json 标签。


---



Figure 12. Prompts for generating contextual information.
图 12. 生成上下文信息的提示。


---



<img>current_screenshot.png</img>
<img>current_screenshot.png</img>


<img>current_screenshot_w_labels.png</img>
<img>current_screenshot_w_labels.png</img>


&nbsp;&nbsp;&nbsp;&nbsp;Based on the original and marked screenshots of an Android mobile phone, where the marked screenshot is the original screenshot marked with action location, please follow the
&nbsp;&nbsp;&nbsp;&nbsp;基于原始截图及其标注的 Android 手机截图，其中标注截图是在原始截图上标注了操作位置，请按照以下说明：


&nbsp;&nbsp;&nbsp;&nbsp;instructions below:
&nbsp;&nbsp;&nbsp;&nbsp;以下说明：


&nbsp;&nbsp;&nbsp;&nbsp;1. Screenshot Description:
&nbsp;&nbsp;&nbsp;&nbsp;1. 截图描述：


&nbsp;&nbsp;&nbsp;&nbsp;- Analyze and describe the overall content of the current screenshot.
&nbsp;&nbsp;&nbsp;&nbsp;- - 分析并描述当前截图的总体内容。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Provide a concise 2-3 sentence summary of the screenshot content.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- - 提供截图内容的简要2-3句概括。


&nbsp;&nbsp;&nbsp;&nbsp;- Format the description as follows: "This is a screenshot of [summary of the screenshot content]."
&nbsp;&nbsp;&nbsp;&nbsp;- - 将描述格式化为： "This is a screenshot of [summary of the screenshot content]." 


&nbsp;&nbsp;&nbsp;&nbsp;2. Intention Recognition:
&nbsp;&nbsp;&nbsp;&nbsp;2. 意图识别：


- You are viewing the current screenshot while completing the task: \{task\} on a mobile phone, and you have chosen to perform the action: \{action\}. Analyze the reasoning behind this
- 在完成任务时你正在查看当前截图：\{task\} 在手机上，并且你选择执行的操作：\{action\}。分析这背后的理由


&nbsp;&nbsp;&nbsp;&nbsp;- The progress made on the task before this action is: \{context\}.
&nbsp;&nbsp;&nbsp;&nbsp;- - 在此操作之前所完成的任务进展：\{context\}。


- Focus on why this action is appropriate within the current context, using present tense as if actively solving the problem.
- 着重说明在当前情境中为何此操作是合适的，使用现在时态，仿佛在主动解决问题。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Explain your intention in the first person.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- - 用第一人称说明你的意图。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Format the intention in 2-3 sentences as follows: "To [goal or purpose], I choose to [action to take]. This allows me to [result or benefit]."
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- - 将意图以2-3句格式如下表达： "To [goal or purpose], I choose to [action to take]. This allows me to [result or benefit]."


P.S.:



&nbsp;&nbsp;&nbsp;&nbsp;- When following these instructions, use natural language, avoid mentioning technical details (such as action coordinates) or direct use of action tags (such as "PRESS_HOME").
&nbsp;&nbsp;&nbsp;&nbsp;- - 遵循这些指示时，使用自然语言，避免提及技术细节（如操作坐标）或直接使用操作标签（如 "PRESS_HOME"）。


&nbsp;&nbsp;&nbsp;&nbsp;Output Format:
&nbsp;&nbsp;&nbsp;&nbsp;输出格式：


The output should be in JSON format as follows:
输出应以如下 JSON 格式：


\{\{



&nbsp;&nbsp;&nbsp;&nbsp;"description": "2-3 sentences summarizing the current screenshot content in the desired format",
&nbsp;&nbsp;&nbsp;&nbsp;"description": "当前屏幕截图内容的 2-3 句摘要，以所需格式"


&nbsp;&nbsp;&nbsp;&nbsp;"intention": "2-3 sentences explaining reasoning for choosing this action in the desired format"
&nbsp;&nbsp;&nbsp;&nbsp;"intention": "解释选择此操作的原因的 2-3 句，以所需格式"


\}\}



Please return the result in pure JSON format, without any json tags like ```json ```.
请以纯 JSON 格式返回结果，不要包含任何 json 标签如 ```json ```。


---



Figure 13. Prompts for generating screen description and decision rationale.
图 13. 生成屏幕描述和决策理由的提示。


Prompt for evaluating closed-source proprietary LVLMs
评估闭源专有 LVLMs 的提示


<img>current_screenshot.png</img>
<img>current_screenshot.png


Given a device screenshot and an instruction, please provide the corresponding action.
给定设备截图与指令，请提供相应的操作。


Available Actions:
可用操作：


CLICK: <coordinate>
点击：<coordinate>


LONG_PRESS: <coordinate>
长按：<coordinate>


TYPE: <text>
输入：<text>


SCROLL: UP
滚动：向上


SCROLL: DOWN
滚动：向下


SCROLL: LEFT
滚动：向左


SCROLL: RIGHT
滚动：向右


PRESS_BACK
返回


PRESS_HOME
按主页


PRESS_RECENT
按最近使用


IMPOSSIBLE
不可能


COMPLETE



All <coordinates> are in the form (x, y), representing the coordinates to click or long press. The coordinate of the top-left corner is $\left( {0,0}\right)$ ,and the coordinate of the bottom-right corner is $\left( {{1000},{1000}}\right)$ .
所有 <coordinates> 的形式为 (x, y)，表示需要点击或长按的坐标。左上角的坐标是 $\left( {0,0}\right)$，右下角的坐标是 $\left( {{1000},{1000}}\right)$ 。


The instuction is: \{instuction\}
指令是：\{instuction\}


The historical actions are: \{history_actions\}
历史动作是：\{history_actions\}


Based on the screenshots and the available actions, provide the next step directly.
基于屏幕截图和可用操作，直接给出下一步。


Prompt for evaluating closed-source proprietary LVLMs with OmniParser
用于评估闭源专有 LVLM 的提示：OmniParser


<img>current_screenshot.png</img>
<img>current_screenshot.png</img>


<img>current_screenshot_w_labels.png</img>
<img>current_screenshot_w_labels.png</img>


Given two device screenshots and an instruction, provide the corresponding action.
给定两个设备屏幕截图和一个指令，给出相应动作。


The first image is the original screenshot, and the second is the same screenshot with numeric tags on different interface elements. If the action requires clicking or pressing, choose the closest numeric tag that aligns with your intended location.
第一张图是原始截图，第二张图是在不同界面元素上标注了数字标签的同一张截图。如果需要点击或按压，请选择与目标位置最接近的数字标签。


Here are the Available Actions:
以下是可用操作：


CLICK: <element_idx chosen from the second screen>
CLICK: <element_idx 从第二屏幕中选择>


LONG_PRESS: <element_idx chosen from the second screen>
LONG_PRESS: <element_idx 从第二屏幕中选择>


TYPE: <text>
TYPE: <text>


SCROLL: UP
向上滚动


SCROLL: DOWN
向下滚动


SCROLL: LEFT
向左滚动


SCROLL: RIGHT
向右滚动


PRESS_BACK
按返回


PRESS_HOME
按主页


PRESS_RECENT
按最近使用


IMPOSSIBLE
不可能


COMPLETE



The instuction is: \{instuction\}
指令是：\{instuction\}


The historical actions are: \{history_actions\}
历史操作为：\{history_actions\}


Based on the screenshots and the available actions, provide the next step directly.
基于屏幕截图和可用操作，直接给出下一步。