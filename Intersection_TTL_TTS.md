# Awesome Learning-and-Scaling Systems [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated paper list covering related training-time amortization, learned control of test-time scaling, and active-deployment joint learning-and-scaling systems.

[← Overview](README.md) · [All papers](ALL_PAPERS.md)

## Contents

- [1. Training-Time Amortization from Inference Compute](#1-training-time-amortization-from-inference-compute)
  - [1.1 Consensus and Self-Generated Targets](#11-consensus-and-self-generated-targets)
  - [1.2 Verifier- and Evaluation-Derived Feedback](#12-verifier--and-evaluation-derived-feedback)
  - [1.3 Trajectory- and Search-Derived Supervision](#13-trajectory--and-search-derived-supervision)
  - [1.4 Search-Gain Distillation](#14-search-gain-distillation)
- [2. Learned Control of Test-Time Scaling](#2-learned-control-of-test-time-scaling)
  - [2.1 Adaptive Budget Allocation](#21-adaptive-budget-allocation)
  - [2.2 Inference-Mechanism Routing](#22-inference-mechanism-routing)
  - [2.3 Model and Cascade Routing](#23-model-and-cascade-routing)
  - [2.4 Learned Scaling Behaviors](#24-learned-scaling-behaviors)
- [3. Active-Deployment Joint Learning-and-Scaling](#3-active-deployment-joint-learning-and-scaling)
  - [3.1 Test-Time Reinforcement Learning](#31-test-time-reinforcement-learning)
    - [3.1.1 Consensus-Derived Rewards](#311-consensus-derived-rewards)
    - [3.1.2 Verified and Stabilized Rewards](#312-verified-and-stabilized-rewards)
    - [3.1.3 Task-Grounded Rewards](#313-task-grounded-rewards)
  - [3.2 Self-Synthesized Supervision](#32-self-synthesized-supervision)
    - [3.2.1 Direct Self-Editing](#321-direct-self-editing)
    - [3.2.2 Instance-Conditioned Synthesis](#322-instance-conditioned-synthesis)
    - [3.2.3 Reflection- and Feedback-Guided Synthesis](#323-reflection--and-feedback-guided-synthesis)

## 1. Training-Time Amortization from Inference Compute

Although their gains arise through later offline training, these methods show how inference-derived signals can become reusable learning targets and thus inform active-deployment joint systems.

### 1.1 Consensus and Self-Generated Targets

- `STaR` [Zelikman et al., NeurIPS 2022] **Star: Bootstrapping reasoning with reasoning** [[Paper]](https://arxiv.org/abs/2203.14465) [[G-Scholar]](https://scholar.google.com/scholar?q=Star%3A%20Bootstrapping%20reasoning%20with%20reasoning) [[CODE]](https://github.com/ezelikman/STaR)
- `ReST-EM` [Singh et al., TMLR 2024] **Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models** [[Paper]](https://arxiv.org/abs/2312.06585) [[G-Scholar]](https://scholar.google.com/scholar?q=Beyond%20Human%20Data%3A%20Scaling%20Self-Training%20for%20Problem-Solving%20with%20Language%20Models)
- `CARE-STaR` [Li et al., Findings of ACL 2025] **CARE-STaR: Constraint-aware Self-taught Reasoner** [[Paper]](https://aclanthology.org/2025.findings-acl.1116/) [[G-Scholar]](https://scholar.google.com/scholar?q=CARE-STaR%3A%20Constraint-aware%20Self-taught%20Reasoner)
- `P-TTS` [Bsharat et al., Findings of ACL 2026] **Prompting Test-Time Scaling Is A Strong LLM Reasoning Data Augmentation** [[Paper]](https://arxiv.org/abs/2510.09599) [[G-Scholar]](https://scholar.google.com/scholar?q=Prompting%20Test-Time%20Scaling%20Is%20A%20Strong%20LLM%20Reasoning%20Data%20Augmentation)

### 1.2 Verifier- and Evaluation-Derived Feedback

- `AutoPSV` [Lu et al., NeurIPS 2024] **Autopsv: Automated process-supervised verifier** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9246aa822579d9b29a140ecdac36ad60-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Autopsv%3A%20Automated%20process-supervised%20verifier)
- [Luo et al., arXiv 2024] **Improve Mathematical Reasoning in Language Models by Automated Process Supervision** [[Paper]](https://arxiv.org/abs/2406.06592) [[G-Scholar]](https://scholar.google.com/scholar?q=Improve%20Mathematical%20Reasoning%20in%20Language%20Models%20by%20Automated%20Process%20Supervision)
- `Math-Shepherd` [Wang et al., ACL 2024] **Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations** [[Paper]](https://aclanthology.org/2024.acl-long.510/) [[G-Scholar]](https://scholar.google.com/scholar?q=Math-Shepherd%3A%20Verify%20and%20Reinforce%20LLMs%20Step-by-step%20without%20Human%20Annotations)
- [Yuan et al., ICML 2024] **Self-Rewarding Language Models** [[Paper]](https://arxiv.org/abs/2401.10020) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-Rewarding%20Language%20Models)
- `DPO-ST` [Wang et al., ACL 2024] **Self-training with direct preference optimization improves chain-of-thought reasoning** [[Paper]](https://aclanthology.org/2024.acl-long.643/) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-training%20with%20direct%20preference%20optimization%20improves%20chain-of-thought%20reasoning)
- `V-STaR` [Hosseini et al., COLM 2024] **V-STaR: Training Verifiers for Self-Taught Reasoners** [[Paper]](https://arxiv.org/abs/2402.06457) [[G-Scholar]](https://scholar.google.com/scholar?q=V-STaR%3A%20Training%20Verifiers%20for%20Self-Taught%20Reasoners)
- `Trainable Self-Verification` [Chen et al., arXiv 2026] **Learning to Self-Verify Makes Language Models Better Reasoners** [[Paper]](https://arxiv.org/abs/2602.07594) [[G-Scholar]](https://scholar.google.com/scholar?q=Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners)

### 1.3 Trajectory- and Search-Derived Supervision

- [Silver et al., Nature 2017] **Mastering the Game of Go without Human Knowledge** [[Record]](https://www.semanticscholar.org/paper/c27db32efa8137cbf654902f8f728f338e55cd1c) [[G-Scholar]](https://scholar.google.com/scholar?q=Mastering%20the%20Game%20of%20Go%20without%20Human%20Knowledge)
- [Anthony et al., NeurIPS 2017] **Thinking Fast and Slow with Deep Learning and Tree Search** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Thinking%20Fast%20and%20Slow%20with%20Deep%20Learning%20and%20Tree%20Search)
- `ReAct` [Yao et al., ICLR 2023] **ReAct: Synergizing Reasoning and Acting in Language Models** [[Paper]](https://openreview.net/forum?id=WE_vluYUL-X) [[G-Scholar]](https://scholar.google.com/scholar?q=ReAct%3A%20Synergizing%20Reasoning%20and%20Acting%20in%20Language%20Models) [[CODE]](https://github.com/ysymyth/ReAct)
- `A³T` [Yang et al., COLM 2024] **ReAct Meets ActRe: When Language Agents Enjoy Training Data Autonomy** [[Paper]](https://arxiv.org/abs/2403.14589) [[G-Scholar]](https://scholar.google.com/scholar?q=ReAct%20Meets%20ActRe%3A%20When%20Language%20Agents%20Enjoy%20Training%20Data%20Autonomy)
- `ReST-MCTS*` [Zhang et al., NeurIPS 2024] **Rest-mcts\*: Llm self-training via process reward guided tree search** [[Paper]](https://arxiv.org/abs/2406.03816) [[G-Scholar]](https://scholar.google.com/scholar?q=Rest-mcts*%3A%20Llm%20self-training%20via%20process%20reward%20guided%20tree%20search)
- `AlphaLLM` [Tian et al., NeurIPS 2024] **Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing** [[Paper]](https://arxiv.org/abs/2404.12253) [[G-Scholar]](https://scholar.google.com/scholar?q=Toward%20Self-Improvement%20of%20LLMs%20via%20Imagination%2C%20Searching%2C%20and%20Criticizing)
- `AlphaLLM-CPL` [Wang et al., arXiv 2024] **Towards Self-Improvement of LLMs via MCTS: Leveraging Stepwise Knowledge with Curriculum Preference Learning** [[Paper]](https://arxiv.org/abs/2410.06508) [[G-Scholar]](https://scholar.google.com/scholar?q=Towards%20Self-Improvement%20of%20LLMs%20via%20MCTS%3A%20Leveraging%20Stepwise%20Knowledge%20with%20Curriculum%20Preference%20Learning)
- `ETO` [Song et al., ACL 2024] **Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents** [[Paper]](https://arxiv.org/abs/2403.02502) [[G-Scholar]](https://scholar.google.com/scholar?q=Trial%20and%20Error%3A%20Exploration-Based%20Trajectory%20Optimization%20for%20LLM%20Agents)
- `Agent-R` [Yuan et al., arXiv 2025] **Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training** [[Paper]](https://arxiv.org/abs/2501.11425) [[G-Scholar]](https://scholar.google.com/scholar?q=Agent-R%3A%20Training%20Language%20Model%20Agents%20to%20Reflect%20via%20Iterative%20Self-Training)
- `rStar-Math` [Guan et al., ICML 2025] **rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking** [[Paper]](https://arxiv.org/abs/2501.04519) [[G-Scholar]](https://scholar.google.com/scholar?q=rStar-Math%3A%20Small%20LLMs%20Can%20Master%20Math%20Reasoning%20with%20Self-Evolved%20Deep%20Thinking)
- `WebRL` [Qi et al., ICLR 2025] **WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning** [[Paper]](https://openreview.net/pdf?id=oVKEAFjEqv) [[G-Scholar]](https://scholar.google.com/scholar?q=WebRL%3A%20Training%20LLM%20Web%20Agents%20via%20Self-Evolving%20Online%20Curriculum%20Reinforcement%20Learning)

### 1.4 Search-Gain Distillation

- `BOND` [Sessa et al., ICLR 2025] **BOND: Aligning LLMs with best-of-n distillation** [[Paper]](https://openreview.net/forum?id=0tAXMiSufG) [[G-Scholar]](https://scholar.google.com/scholar?q=BOND%3A%20Aligning%20LLMs%20with%20best-of-n%20distillation)
- `Fast Quiet-STaR` [Huang et al., Findings of EMNLP 2025] **Fast Quiet-STaR: Thinking Without Thought Tokens** [[Paper]](https://aclanthology.org/2025.findings-emnlp.1020/) [[G-Scholar]](https://scholar.google.com/scholar?q=Fast%20Quiet-STaR%3A%20Thinking%20Without%20Thought%20Tokens)
- `Faster WIND` [Yang et al., AISTATS 2025] **Faster WIND: Accelerating Iterative Best-of-N Distillation for LLM Alignment** [[Paper]](https://proceedings.mlr.press/v258/yang25e.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Faster%20WIND%3A%20Accelerating%20Iterative%20Best-of-N%20Distillation%20for%20LLM%20Alignment)

## 2. Learned Control of Test-Time Scaling

Although most controllers are learned before deployment, they show how compute allocation and routing can be coupled with online updates in future joint systems.

### 2.1 Adaptive Budget Allocation

- [Graves, arXiv 2016] **Adaptive Computation Time for Recurrent Neural Networks** [[Paper]](https://arxiv.org/abs/1603.08983) [[G-Scholar]](https://scholar.google.com/scholar?q=Adaptive%20Computation%20Time%20for%20Recurrent%20Neural%20Networks)
- [Schuster et al., NeurIPS 2022] **Confident Adaptive Language Modeling** [[Paper]](https://arxiv.org/abs/2207.07061) [[G-Scholar]](https://scholar.google.com/scholar?q=Confident%20Adaptive%20Language%20Modeling)
- [Damani et al., ICLR 2025] **Learning How Hard to Think: Input-Adaptive Allocation of LM Computation** [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ff414825df833edb8b1839e3d5d495e9-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Learning%20How%20Hard%20to%20Think%3A%20Input-Adaptive%20Allocation%20of%20LM%20Computation)
- `TALE` [Han et al., Findings of ACL 2025] **Token-budget-aware LLM reasoning** [[Paper]](https://aclanthology.org/2025.findings-acl.1274/) [[G-Scholar]](https://scholar.google.com/scholar?q=Token-budget-aware%20LLM%20reasoning)
- [Xiong et al., arXiv 2026] **Learning When to Sample: Confidence-Aware Selective Sampling for Efficient Chain-of-Thought Reasoning** [[Paper]](https://arxiv.org/abs/2603.08999) [[G-Scholar]](https://scholar.google.com/scholar?q=Learning%20When%20to%20Sample%3A%20Confidence-Aware%20Selective%20Sampling%20for%20Efficient%20Chain-of-Thought%20Reasoning)
- `TAB` [Jali et al., arXiv 2026] **Not All Turns Are Equally Hard: Adaptive Thinking Budgets For Efficient Multi-Turn Reasoning** [[Paper]](https://arxiv.org/abs/2604.05164) [[G-Scholar]](https://scholar.google.com/scholar?q=Not%20All%20Turns%20Are%20Equally%20Hard%3A%20Adaptive%20Thinking%20Budgets%20For%20Efficient%20Multi-Turn%20Reasoning)
- `SelfBudgeter` [Li et al., Findings of ACL 2026] **SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning** [[Paper]](https://arxiv.org/abs/2505.11274) [[G-Scholar]](https://scholar.google.com/scholar?q=SelfBudgeter%3A%20Adaptive%20Token%20Allocation%20for%20Efficient%20LLM%20Reasoning)
- [Zuo et al., ICLR 2026] **Strategic Scaling of Test-Time Compute: A Bandit Learning Approach** [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8d2c5f79c53a2a685dbdf93f78c5695-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Strategic%20Scaling%20of%20Test-Time%20Compute%3A%20A%20Bandit%20Learning%20Approach)

### 2.2 Inference-Mechanism Routing

- `Toolformer` [Schick et al., NeurIPS 2023] **Toolformer: Language Models Can Teach Themselves to Use Tools** [[Paper]](https://arxiv.org/abs/2302.04761) [[G-Scholar]](https://scholar.google.com/scholar?q=Toolformer%3A%20Language%20Models%20Can%20Teach%20Themselves%20to%20Use%20Tools)
- `Adaptive-RAG` [Jeong et al., ACL 2024] **Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity** [[Paper]](https://arxiv.org/abs/2403.14403) [[G-Scholar]](https://scholar.google.com/scholar?q=Adaptive-rag%3A%20Learning%20to%20adapt%20retrieval-augmented%20large%20language%20models%20through%20question%20complexity) [[CODE]](https://github.com/starsuzi/Adaptive-RAG)
- `Self-RAG` [Asai et al., ICLR 2024] **Self-RAG: Learning to retrieve, generate, and critique through self-reflection** [[Paper]](https://openreview.net/forum?id=hSyW5go0v8) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-RAG%3A%20Learning%20to%20retrieve%2C%20generate%2C%20and%20critique%20through%20self-reflection) [[CODE]](https://github.com/akariasai/self-rag)
- `AdaptThink` [Zhang et al., EMNLP 2025] **Adaptthink: Reasoning models can learn when to think** [[Paper]](https://arxiv.org/abs/2505.13417) [[G-Scholar]](https://scholar.google.com/scholar?q=Adaptthink%3A%20Reasoning%20models%20can%20learn%20when%20to%20think)
- `Adaptive Parallel Reasoning` [Pan et al., COLM 2025] **Learning Adaptive Parallel Reasoning with Language Models** [[Paper]](https://openreview.net/forum?id=rOKV2ubxgV) [[G-Scholar]](https://scholar.google.com/scholar?q=Learning%20Adaptive%20Parallel%20Reasoning%20with%20Language%20Models)
- [Paglieri et al., arXiv 2025] **Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents** [[Paper]](https://arxiv.org/abs/2509.03581) [[G-Scholar]](https://scholar.google.com/scholar?q=Learning%20When%20to%20Plan%3A%20Efficiently%20Allocating%20Test-Time%20Compute%20for%20LLM%20Agents)
- `RouteRAG` [Guo et al., Findings of ACL 2026] **RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning** [[Paper]](https://aclanthology.org/2026.findings-acl.1502/) [[G-Scholar]](https://scholar.google.com/scholar?q=RouteRAG%3A%20Efficient%20Retrieval-Augmented%20Generation%20from%20Text%20and%20Graph%20via%20Reinforcement%20Learning)

### 2.3 Model and Cascade Routing

- `FrugalGPT` [Chen et al., TMLR 2024] **FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance** [[Paper]](https://arxiv.org/abs/2305.05176) [[G-Scholar]](https://scholar.google.com/scholar?q=FrugalGPT%3A%20How%20to%20Use%20Large%20Language%20Models%20While%20Reducing%20Cost%20and%20Improving%20Performance)
- `RouteLLM` [Ong et al., ICLR 2025] **RouteLLM: Learning to Route LLMs from Preference Data** [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=RouteLLM%3A%20Learning%20to%20Route%20LLMs%20from%20Preference%20Data)

### 2.4 Learned Scaling Behaviors

- `Satori` [Shen et al., ICML 2025] **Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search** [[Paper]](https://arxiv.org/abs/2502.02508) [[G-Scholar]](https://scholar.google.com/scholar?q=Satori%3A%20Reinforcement%20Learning%20with%20Chain-of-Action-Thought%20Enhances%20LLM%20Reasoning%20via%20Autoregressive%20Search)
- [Kumar et al., ICLR 2025] **Training Language Models to Self-Correct via Reinforcement Learning** [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2025/hash/871ac99fdc5282d0301934d23945ebaa-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Training%20Language%20Models%20to%20Self-Correct%20via%20Reinforcement%20Learning)

## 3. Active-Deployment Joint Learning-and-Scaling

### 3.1 Test-Time Reinforcement Learning

#### 3.1.1 Consensus-Derived Rewards

- `TTRL` [Zuo et al., NeurIPS 2025] **TTRL: Test-Time Reinforcement Learning** [[Paper]](https://arxiv.org/abs/2504.16084) [[G-Scholar]](https://scholar.google.com/scholar?q=TTRL%3A%20Test-Time%20Reinforcement%20Learning) [[CODE]](https://github.com/PRIME-RL/TTRL)
- `TTRV` [Singh et al., arXiv 2025] **TTRV: Test-Time Reinforcement Learning for Vision Language Models** [[Paper]](https://arxiv.org/abs/2510.06783) [[G-Scholar]](https://scholar.google.com/scholar?q=TTRV%3A%20Test-Time%20Reinforcement%20Learning%20for%20Vision%20Language%20Models)

#### 3.1.2 Verified and Stabilized Rewards

- `SCOPE` [Wang et al., ACL 2026] **Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning** [[Paper]](https://arxiv.org/abs/2512.15146) [[G-Scholar]](https://scholar.google.com/scholar?q=Beyond%20Majority%20Voting%3A%20Towards%20Fine-grained%20and%20More%20Reliable%20Reward%20Signal%20for%20Test-Time%20Reinforcement%20Learning)
- `DARE` [Du et al., arXiv 2026] **Distribution-Aware Reward Estimation for Test-Time Reinforcement Learning** [[Paper]](https://arxiv.org/abs/2601.21804) [[G-Scholar]](https://scholar.google.com/scholar?q=Distribution-Aware%20Reward%20Estimation%20for%20Test-Time%20Reinforcement%20Learning)
- `T3RL` [Liao et al., arXiv 2026] **Tool Verification for Test-Time Reinforcement Learning** [[Paper]](https://arxiv.org/abs/2603.02203) [[G-Scholar]](https://scholar.google.com/scholar?q=Tool%20Verification%20for%20Test-Time%20Reinforcement%20Learning)
- `DDRL` [Yu et al., Findings of ACL 2026] **Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning** [[Paper]](https://arxiv.org/abs/2604.21327) [[G-Scholar]](https://scholar.google.com/scholar?q=Understanding%20and%20Mitigating%20Spurious%20Signal%20Amplification%20in%20Test-Time%20Reinforcement%20Learning%20for%20Math%20Reasoning)

#### 3.1.3 Task-Grounded Rewards

- `Alpha-RTL` [Zhou et al., arXiv 2026] **Alpha-RTL: Test-Time Training for RTL Hardware Optimization** [[Paper]](https://arxiv.org/abs/2606.05253) [[G-Scholar]](https://scholar.google.com/scholar?q=Alpha-RTL%3A%20Test-Time%20Training%20for%20RTL%20Hardware%20Optimization)
- `TTT-Discover` [Yuksekgonul et al., arXiv 2026] **Learning to Discover at Test Time** [[Paper]](https://arxiv.org/abs/2601.16175) [[G-Scholar]](https://scholar.google.com/scholar?q=Learning%20to%20Discover%20at%20Test%20Time)

### 3.2 Self-Synthesized Supervision

#### 3.2.1 Direct Self-Editing

- `SEAL` [Zweiger et al., NeurIPS 2025] **Self-Adapting Language Models** [[Paper]](https://arxiv.org/abs/2506.10943) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-Adapting%20Language%20Models) [[CODE]](https://github.com/Continual-Intelligence/SEAL)

#### 3.2.2 Instance-Conditioned Synthesis

- `QueST` [Song et al., arXiv 2026] **Query-Conditioned Test-Time Self-Training for Large Language Models** [[Paper]](https://arxiv.org/abs/2605.13369) [[G-Scholar]](https://scholar.google.com/scholar?q=Query-Conditioned%20Test-Time%20Self-Training%20for%20Large%20Language%20Models)
- `MASS` [Kaya et al., arXiv 2026] **Test-Time Meta-Adaptation with Self-Synthesis** [[Paper]](https://arxiv.org/abs/2603.03524) [[G-Scholar]](https://scholar.google.com/scholar?q=Test-Time%20Meta-Adaptation%20with%20Self-Synthesis)
- `TT-SI` [Acikgoz et al., Findings of ACL 2026] **TT-SI: Self-Improving LLM Agents with Test-Time Training** [[Paper]](https://aclanthology.org/2026.findings-acl.462/) [[G-Scholar]](https://scholar.google.com/scholar?q=TT-SI%3A%20Self-Improving%20LLM%20Agents%20with%20Test-Time%20Training)

#### 3.2.3 Reflection- and Feedback-Guided Synthesis

- `Test-Time Self-Distillation` [Hubotter et al., ICLR Workshop 2026] **Test-Time Self-Distillation** [[Paper]](https://openreview.net/pdf?id=iEmRSwdzyw) [[G-Scholar]](https://scholar.google.com/scholar?q=Test-Time%20Self-Distillation)
- `TTCS` [Yang et al., arXiv 2026] **TTCS: Test-Time Curriculum Synthesis for Self-Evolving** [[Paper]](https://arxiv.org/abs/2601.22628) [[G-Scholar]](https://scholar.google.com/scholar?q=TTCS%3A%20Test-Time%20Curriculum%20Synthesis%20for%20Self-Evolving)
- `TTSR` [He et al., arXiv 2026] **TTSR: Test-Time Self-Evolving via Reflection** [[Paper]](https://arxiv.org/abs/2603.03297) [[G-Scholar]](https://scholar.google.com/scholar?q=TTSR%3A%20Test-Time%20Self-Evolving%20via%20Reflection)
