# Awesome Test-Time Scaling [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated paper list for Test-Time Scaling (TTS), organized by internal inference compute, external resources, and supporting theory and evaluation.

[← Overview](README.md) · [All papers](ALL_PAPERS.md)

## Contents

- [1. Scaling Internal Inference Compute](#1-scaling-internal-inference-compute)
  - [1.1 Sequential and Structured Reasoning](#11-sequential-and-structured-reasoning)
  - [1.2 Iterative Refinement and Latent Computation](#12-iterative-refinement-and-latent-computation)
  - [1.3 Parallel Sampling and Candidate Selection](#13-parallel-sampling-and-candidate-selection)
  - [1.4 Consensus and Soft Aggregation](#14-consensus-and-soft-aggregation)
  - [1.5 Search-Based Inference and Planning](#15-search-based-inference-and-planning)
  - [1.6 Adaptive Compute Allocation](#16-adaptive-compute-allocation)
- [2. Scaling with External Resources](#2-scaling-with-external-resources)
  - [2.1 Adaptive Retrieval and Search-Augmented Reasoning](#21-adaptive-retrieval-and-search-augmented-reasoning)
  - [2.2 Tool Use and Program Execution](#22-tool-use-and-program-execution)
  - [2.3 Verification, Critique, and Re-Ranking](#23-verification-critique-and-re-ranking)
  - [2.4 Multi-Agent Inference and Collaboration](#24-multi-agent-inference-and-collaboration)
- [3. Theory and Evaluation of Test-Time Scaling](#3-theory-and-evaluation-of-test-time-scaling)
  - [3.1 Sampling, Verification, and Scaling Laws](#31-sampling-verification-and-scaling-laws)
  - [3.2 Compute Allocation and Diminishing Returns](#32-compute-allocation-and-diminishing-returns)
  - [3.3 Reliability, Safety, and Adversarial Evaluation](#33-reliability-safety-and-adversarial-evaluation)
  - [3.4 Persistence and Amortization Boundaries](#34-persistence-and-amortization-boundaries)

## 1. Scaling Internal Inference Compute

### 1.1 Sequential and Structured Reasoning

- `Chain-of-Thought` [Wei et al., NeurIPS 2022] **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** [[Paper]](https://arxiv.org/abs/2201.11903) [[G-Scholar]](https://scholar.google.com/scholar?q=Chain-of-Thought%20Prompting%20Elicits%20Reasoning%20in%20Large%20Language%20Models)
- `Least-to-Most` [Zhou et al., ICLR 2023] **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models** [[Paper]](https://arxiv.org/abs/2205.10625) [[G-Scholar]](https://scholar.google.com/scholar?q=Least-to-Most%20Prompting%20Enables%20Complex%20Reasoning%20in%20Large%20Language%20Models)
- `Algorithm of Thoughts` [Sel et al., ICML 2024] **Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models** [[Paper]](https://arxiv.org/abs/2308.10379) [[G-Scholar]](https://scholar.google.com/scholar?q=Algorithm%20of%20Thoughts%3A%20Enhancing%20Exploration%20of%20Ideas%20in%20Large%20Language%20Models)
- `SELF-DISCOVER` [Zhou et al., arXiv 2024] **SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures** [[Paper]](https://arxiv.org/abs/2402.03620) [[G-Scholar]](https://scholar.google.com/scholar?q=SELF-DISCOVER%3A%20Large%20Language%20Models%20Self-Compose%20Reasoning%20Structures)

### 1.2 Iterative Refinement and Latent Computation

- `Self-Refine` [Madaan et al., NeurIPS 2023] **Self-refine: Iterative refinement with self-feedback** [[Paper]](https://arxiv.org/abs/2303.17651) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-refine%3A%20Iterative%20refinement%20with%20self-feedback)
- `Hidden Computation` [Pfau et al., COLM 2024] **Lets Think Dot by Dot: Hidden computation in transformer language models** [[Paper]](https://arxiv.org/abs/2404.15758) [[G-Scholar]](https://scholar.google.com/scholar?q=Lets%20Think%20Dot%20by%20Dot%3A%20Hidden%20computation%20in%20transformer%20language%20models)
- `Looped Transformers` [Saunshi et al., ICLR 2025] **Reasoning with Latent Thoughts: On the Power of Looped Transformers** [[Paper]](https://openreview.net/forum?id=6Nq6N9JkBe) [[G-Scholar]](https://scholar.google.com/scholar?q=Reasoning%20with%20Latent%20Thoughts%3A%20On%20the%20Power%20of%20Looped%20Transformers)
- `Recurrent Depth` [Geiping et al., NeurIPS 2025] **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach** [[Paper]](https://arxiv.org/abs/2502.05171) [[G-Scholar]](https://scholar.google.com/scholar?q=Scaling%20up%20Test-Time%20Compute%20with%20Latent%20Reasoning%3A%20A%20Recurrent%20Depth%20Approach) [[CODE]](https://github.com/seal-rg/recurrent-pretraining)

### 1.3 Parallel Sampling and Candidate Selection

- `AlphaCode` [Li et al., Science 2022] **Competition-Level Code Generation with AlphaCode** [[Paper]](https://arxiv.org/abs/2203.07814) [[G-Scholar]](https://scholar.google.com/scholar?q=Competition-Level%20Code%20Generation%20with%20AlphaCode)
- `Speculative Rejection` [Sun et al., NeurIPS 2024] **Fast best-of-n decoding via speculative rejection** [[Paper]](https://papers.nips.cc/paper_files/paper/2024/hash/3950f6bf5c2eb7435ecf58eaa85cc8c2-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Fast%20best-of-n%20decoding%20via%20speculative%20rejection)
- [Brown et al., arXiv 2024] **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** [[Paper]](https://arxiv.org/abs/2407.21787) [[G-Scholar]](https://scholar.google.com/scholar?q=Large%20Language%20Monkeys%3A%20Scaling%20Inference%20Compute%20with%20Repeated%20Sampling)
- `MBR-BoN` [Jinnai et al., NAACL 2025] **Regularized best-of-n sampling with minimum bayes risk objective for language model alignment** [[Paper]](https://aclanthology.org/2025.naacl-long.472/) [[G-Scholar]](https://scholar.google.com/scholar?q=Regularized%20best-of-n%20sampling%20with%20minimum%20bayes%20risk%20objective%20for%20language%20model%20alignment)
- `Self-Certainty` [Kang et al., NeurIPS 2025] **Scalable Best-of-N Selection for Large Language Models via Self-Certainty** [[Paper]](https://proceedings.nips.cc/paper_files/paper/2025/hash/1c7eff166a8e345f664f0faa8f4e4d2e-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Scalable%20Best-of-N%20Selection%20for%20Large%20Language%20Models%20via%20Self-Certainty)

### 1.4 Consensus and Soft Aggregation

- `Self-Consistency` [Wang et al., ICLR 2023] **Self-Consistency Improves Chain of Thought Reasoning in Language Models** [[Paper]](https://arxiv.org/abs/2203.11171) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-Consistency%20Improves%20Chain%20of%20Thought%20Reasoning%20in%20Language%20Models)
- `Representation-Agreement Calibration` [Xie et al., NeurIPS 2024] **Calibrating reasoning in language models with internal consistency** [[Paper]](https://arxiv.org/abs/2405.18711) [[G-Scholar]](https://scholar.google.com/scholar?q=Calibrating%20reasoning%20in%20language%20models%20with%20internal%20consistency)
- [Wang et al., arXiv 2024] **Soft Self-Consistency Improves Language Model Agents** [[Paper]](https://arxiv.org/abs/2402.13212) [[G-Scholar]](https://scholar.google.com/scholar?q=Soft%20Self-Consistency%20Improves%20Language%20Model%20Agents) [[CODE]](https://github.com/HanNight/soft_self_consistency)
- `CISC` [Taubenfeld et al., Findings of ACL 2025] **Confidence Improves Self-Consistency in LLMs** [[Paper]](https://arxiv.org/abs/2502.06233) [[G-Scholar]](https://scholar.google.com/scholar?q=Confidence%20Improves%20Self-Consistency%20in%20LLMs)

### 1.5 Search-Based Inference and Planning

- [Hao et al., EMNLP 2023] **Reasoning with Language Model is Planning with World Model** [[Paper]](https://aclanthology.org/2023.emnlp-main.507/) [[G-Scholar]](https://scholar.google.com/scholar?q=Reasoning%20with%20Language%20Model%20is%20Planning%20with%20World%20Model)
- `ToT` [Yao et al., NeurIPS 2023] **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** [[Paper]](https://arxiv.org/abs/2305.10601) [[G-Scholar]](https://scholar.google.com/scholar?q=Tree%20of%20Thoughts%3A%20Deliberate%20Problem%20Solving%20with%20Large%20Language%20Models)
- `ADaPT` [Prasad et al., Findings of ACL 2024] **ADaPT: As-Needed Decomposition and Planning with Language Models** [[Paper]](https://arxiv.org/abs/2311.05772) [[G-Scholar]](https://scholar.google.com/scholar?q=ADaPT%3A%20As-Needed%20Decomposition%20and%20Planning%20with%20Language%20Models)
- `AlphaMath` [Chen et al., NeurIPS 2024] **Alphamath almost zero: process supervision without process** [[Paper]](https://arxiv.org/abs/2405.03553) [[G-Scholar]](https://scholar.google.com/scholar?q=Alphamath%20almost%20zero%3A%20process%20supervision%20without%20process)
- `TS-LLM` [Wan et al., ICML 2024] **AlphaZero-Like Tree-Search can Guide Large Language Model Decoding and Training** [[Paper]](https://arxiv.org/abs/2309.17179) [[G-Scholar]](https://scholar.google.com/scholar?q=AlphaZero-Like%20Tree-Search%20can%20Guide%20Large%20Language%20Model%20Decoding%20and%20Training)
- `Graph-of-Thought` [Besta et al., AAAI 2024] **Graph of thoughts: Solving elaborate problems with large language models** [[Paper]](https://arxiv.org/abs/2308.09687) [[G-Scholar]](https://scholar.google.com/scholar?q=Graph%20of%20thoughts%3A%20Solving%20elaborate%20problems%20with%20large%20language%20models)
- `LATS` [Zhou et al., ICML 2024] **Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models** [[Paper]](https://proceedings.mlr.press/v235/zhou24r.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Language%20Agent%20Tree%20Search%20Unifies%20Reasoning%2C%20Acting%2C%20and%20Planning%20in%20Language%20Models) [[CODE]](https://github.com/lapisrocks/LanguageAgentTreeSearch)
- `ETS` [Hooper et al., arXiv 2025] **ETS: Efficient tree search for inference-time scaling** [[Paper]](https://arxiv.org/abs/2502.13575) [[G-Scholar]](https://scholar.google.com/scholar?q=ETS%3A%20Efficient%20tree%20search%20for%20inference-time%20scaling)
- `World-Model Planning` [Yu et al., TMLR 2025] **Generating Symbolic World Models via Test-time Scaling of Large Language Models** [[Paper]](https://openreview.net/forum?id=zVo6PfBa0K) [[G-Scholar]](https://scholar.google.com/scholar?q=Generating%20Symbolic%20World%20Models%20via%20Test-time%20Scaling%20of%20Large%20Language%20Models)
- `Plan-and-Act` [Erdogan et al., ICML 2025] **Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks** [[Paper]](https://arxiv.org/abs/2503.09572) [[G-Scholar]](https://scholar.google.com/scholar?q=Plan-and-Act%3A%20Improving%20Planning%20of%20Agents%20for%20Long-Horizon%20Tasks)
- `ARMAP` [Chen et al., ICLR 2025] **Scaling Autonomous Agents via Automatic Reward Modeling And Planning** [[Paper]](https://arxiv.org/abs/2502.12130) [[G-Scholar]](https://scholar.google.com/scholar?q=Scaling%20Autonomous%20Agents%20via%20Automatic%20Reward%20Modeling%20And%20Planning)
- `CMCTS` [Lin et al., Applied Intelligence 2026] **CMCTS: A Constrained Monte Carlo Tree Search framework for mathematical reasoning in large language model** [[Paper]](https://doi.org/10.1007/s10489-025-07044-6) [[G-Scholar]](https://scholar.google.com/scholar?q=CMCTS%3A%20A%20Constrained%20Monte%20Carlo%20Tree%20Search%20framework%20for%20mathematical%20reasoning%20in%20large%20language%20model)
- `Interaction Scaling` [Shen et al., NeurIPS 2026] **Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction** [[Paper]](https://arxiv.org/abs/2506.07976) [[G-Scholar]](https://scholar.google.com/scholar?q=Thinking%20vs.%20Doing%3A%20Agents%20that%20Reason%20by%20Scaling%20Test-Time%20Interaction)

### 1.6 Adaptive Compute Allocation

- `Compute-Optimal TTS` [Liu et al., ICLR Workshop 2025] **Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling** [[Paper]](https://arxiv.org/abs/2502.06703) [[G-Scholar]](https://scholar.google.com/scholar?q=Can%201B%20LLM%20Surpass%20405B%20LLM%3F%20Rethinking%20Compute-Optimal%20Test-Time%20Scaling)
- `Efficient Confidence-Aware Scaling` [Huang et al., NeurIPS Workshop 2025] **Efficient Test-Time Scaling via Self-Calibration** [[Paper]](https://arxiv.org/abs/2503.00031) [[G-Scholar]](https://scholar.google.com/scholar?q=Efficient%20Test-Time%20Scaling%20via%20Self-Calibration)
- `LCPO` [Aggarwal et al., COLM 2025] **L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning** [[Paper]](https://openreview.net/forum?id=4jdIxXBNve) [[G-Scholar]](https://scholar.google.com/scholar?q=L1%3A%20Controlling%20How%20Long%20A%20Reasoning%20Model%20Thinks%20With%20Reinforcement%20Learning)
- `DSC` [Wang et al., Findings of NAACL 2025] **Make every penny count: Difficulty-adaptive self-consistency for cost-efficient reasoning** [[Paper]](https://aclanthology.org/2025.findings-naacl.383/) [[G-Scholar]](https://scholar.google.com/scholar?q=Make%20every%20penny%20count%3A%20Difficulty-adaptive%20self-consistency%20for%20cost-efficient%20reasoning)
- `Meta-Reasoner` [Sui et al., arXiv 2025] **Meta-reasoner: Dynamic guidance for optimized inference-time reasoning in large language models** [[Paper]](https://arxiv.org/abs/2502.19918) [[G-Scholar]](https://scholar.google.com/scholar?q=Meta-reasoner%3A%20Dynamic%20guidance%20for%20optimized%20inference-time%20reasoning%20in%20large%20language%20models)
- `s1` [Muennighoff et al., arXiv 2025] **s1: Simple Test-Time Scaling** [[Paper]](https://arxiv.org/abs/2501.19393) [[G-Scholar]](https://scholar.google.com/scholar?q=s1%3A%20Simple%20Test-Time%20Scaling)
- [Snell et al., ICLR 2025] **Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning** [[Paper]](https://arxiv.org/abs/2408.03314) [[G-Scholar]](https://scholar.google.com/scholar?q=Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning)
- `TALE` [Han et al., Findings of ACL 2025] **Token-budget-aware LLM reasoning** [[Paper]](https://aclanthology.org/2025.findings-acl.1274/) [[G-Scholar]](https://scholar.google.com/scholar?q=Token-budget-aware%20LLM%20reasoning)
- `CATTS` [Lee et al., arXiv 2026] **Agentic Test-Time Scaling for WebAgents** [[Paper]](https://arxiv.org/abs/2602.12276) [[G-Scholar]](https://scholar.google.com/scholar?q=Agentic%20Test-Time%20Scaling%20for%20WebAgents)
- `Plan and Budget` [Lin et al., ICLR 2026] **Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning Large Language Models** [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ae8d4084f418bb51575c2ca6c658a05b-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Plan%20and%20Budget%3A%20Effective%20and%20Efficient%20Test-Time%20Scaling%20on%20Reasoning%20Large%20Language%20Models)

## 2. Scaling with External Resources

### 2.1 Adaptive Retrieval and Search-Augmented Reasoning

- `FLARE` [Jiang et al., arXiv 2023] **Active Retrieval Augmented Generation** [[Paper]](https://arxiv.org/abs/2305.06983) [[G-Scholar]](https://scholar.google.com/scholar?q=Active%20Retrieval%20Augmented%20Generation) [[CODE]](https://github.com/jzbjyb/FLARE)
- `IRCoT` [Trivedi et al., ACL 2023] **Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions** [[Paper]](https://aclanthology.org/2023.acl-long.557/) [[G-Scholar]](https://scholar.google.com/scholar?q=Interleaving%20retrieval%20with%20chain-of-thought%20reasoning%20for%20knowledge-intensive%20multi-step%20questions)
- `Adaptive-RAG` [Jeong et al., ACL 2024] **Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity** [[Paper]](https://arxiv.org/abs/2403.14403) [[G-Scholar]](https://scholar.google.com/scholar?q=Adaptive-rag%3A%20Learning%20to%20adapt%20retrieval-augmented%20large%20language%20models%20through%20question%20complexity) [[CODE]](https://github.com/starsuzi/Adaptive-RAG)
- `Self-RAG` [Asai et al., ICLR 2024] **Self-RAG: Learning to retrieve, generate, and critique through self-reflection** [[Paper]](https://openreview.net/forum?id=hSyW5go0v8) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-RAG%3A%20Learning%20to%20retrieve%2C%20generate%2C%20and%20critique%20through%20self-reflection)
- `Search-o1` [Li et al., arXiv 2025] **Search-o1: Agentic Search-Enhanced Large Reasoning Models** [[Paper]](https://arxiv.org/abs/2501.05366) [[G-Scholar]](https://scholar.google.com/scholar?q=Search-o1%3A%20Agentic%20Search-Enhanced%20Large%20Reasoning%20Models) [[CODE]](https://github.com/RUC-NLPIR/Search-o1)
- `Search-R1` [Jin et al., arXiv 2025] **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** [[Paper]](https://arxiv.org/abs/2503.09516) [[G-Scholar]](https://scholar.google.com/scholar?q=Search-R1%3A%20Training%20LLMs%20to%20Reason%20and%20Leverage%20Search%20Engines%20with%20Reinforcement%20Learning) [[CODE]](https://github.com/PeterGriffinJin/Search-R1)
- `MIRAGE` [Wei et al., AAAI 2026] **MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/40673) [[G-Scholar]](https://scholar.google.com/scholar?q=MIRAGE%3A%20Scaling%20Test-Time%20Inference%20with%20Parallel%20Graph-Retrieval-Augmented%20Reasoning%20Chains)

### 2.2 Tool Use and Program Execution

- `XoT` [Liu et al., EMNLP 2023] **Plan, verify and switch: Integrated reasoning with diverse x-of-thoughts** [[Paper]](https://aclanthology.org/2023.emnlp-main.169/) [[G-Scholar]](https://scholar.google.com/scholar?q=Plan%2C%20verify%20and%20switch%3A%20Integrated%20reasoning%20with%20diverse%20x-of-thoughts)
- `PoT` [Chen et al., TMLR 2023] **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks** [[Paper]](https://arxiv.org/abs/2211.12588) [[G-Scholar]](https://scholar.google.com/scholar?q=Program%20of%20Thoughts%20Prompting%3A%20Disentangling%20Computation%20from%20Reasoning%20for%20Numerical%20Reasoning%20Tasks) [[CODE]](https://github.com/TIGER-AI-Lab/Program-of-Thoughts)
- `ReAct` [Yao et al., ICLR 2023] **ReAct: Synergizing Reasoning and Acting in Language Models** [[Paper]](https://openreview.net/forum?id=WE_vluYUL-X) [[G-Scholar]](https://scholar.google.com/scholar?q=ReAct%3A%20Synergizing%20Reasoning%20and%20Acting%20in%20Language%20Models)
- `CRITIC` [Gou et al., ICLR 2024] **Critic: Large language models can self-correct with tool-interactive critiquing** [[Paper]](https://arxiv.org/abs/2305.11738) [[G-Scholar]](https://scholar.google.com/scholar?q=Critic%3A%20Large%20language%20models%20can%20self-correct%20with%20tool-interactive%20critiquing) [[CODE]](https://github.com/microsoft/ProphetNet/tree/master/CRITIC)
- `START` [Li et al., EMNLP 2025] **Start: Self-taught reasoner with tools** [[Paper]](https://arxiv.org/abs/2503.04625) [[G-Scholar]](https://scholar.google.com/scholar?q=Start%3A%20Self-taught%20reasoner%20with%20tools)
- `TTE` [Lu et al., arXiv 2026] **Beyond Static Tools: Test-Time Tool Evolution for Scientific Reasoning** [[Paper]](https://arxiv.org/abs/2601.07641) [[G-Scholar]](https://scholar.google.com/scholar?q=Beyond%20Static%20Tools%3A%20Test-Time%20Tool%20Evolution%20for%20Scientific%20Reasoning)

### 2.3 Verification, Critique, and Re-Ranking

- [Cobbe et al., arXiv 2021] **Training Verifiers to Solve Math Word Problems** [[Paper]](https://arxiv.org/abs/2110.14168) [[G-Scholar]](https://scholar.google.com/scholar?q=Training%20Verifiers%20to%20Solve%20Math%20Word%20Problems)
- `CLoud` [Ankner et al., NeurIPS Workshop 2024] **Critique-out-Loud Reward Models** [[Paper]](https://arxiv.org/abs/2408.11791) [[G-Scholar]](https://scholar.google.com/scholar?q=Critique-out-Loud%20Reward%20Models)
- [Lightman et al., ICLR 2024] **Let's verify step by step** [[Paper]](https://arxiv.org/abs/2305.20050) [[G-Scholar]](https://scholar.google.com/scholar?q=Let's%20verify%20step%20by%20step)
- `MiPS` [Wang et al., Findings of EMNLP 2024] **Multi-step Problem Solving Through a Verifier: An Empirical Analysis on Model-induced Process Supervision** [[Paper]](https://aclanthology.org/2024.findings-emnlp.429/) [[G-Scholar]](https://scholar.google.com/scholar?q=Multi-step%20Problem%20Solving%20Through%20a%20Verifier%3A%20An%20Empirical%20Analysis%20on%20Model-induced%20Process%20Supervision)
- `GenRM` [Zhang et al., ICLR 2025] **Generative verifiers: Reward modeling as next-token prediction** [[Paper]](https://arxiv.org/abs/2408.15240) [[G-Scholar]](https://scholar.google.com/scholar?q=Generative%20verifiers%3A%20Reward%20modeling%20as%20next-token%20prediction)
- `MAV` [Lifshitz et al., ICLR Workshop 2025] **Multi-Agent Verification: Scaling Test-Time Compute with Goal Verifiers** [[Paper]](https://openreview.net/forum?id=LriQ3NY9uL) [[G-Scholar]](https://scholar.google.com/scholar?q=Multi-Agent%20Verification%3A%20Scaling%20Test-Time%20Compute%20with%20Goal%20Verifiers)
- `PairJudge RM` [Liu et al., arXiv 2025] **Pairjudge RM: Perform best-of-N sampling with knockout tournament** [[Paper]](https://arxiv.org/abs/2501.13007) [[G-Scholar]](https://scholar.google.com/scholar?q=Pairjudge%20RM%3A%20Perform%20best-of-N%20sampling%20with%20knockout%20tournament)

### 2.4 Multi-Agent Inference and Collaboration

- `MAD` [Liang et al., EMNLP 2024] **Encouraging divergent thinking in large language models through multi-agent debate** [[Paper]](https://aclanthology.org/2024.emnlp-main.992/) [[G-Scholar]](https://scholar.google.com/scholar?q=Encouraging%20divergent%20thinking%20in%20large%20language%20models%20through%20multi-agent%20debate)
- `Agent Forest` [li et al., TMLR 2024] **More Agents Is All You Need** [[Paper]](https://arxiv.org/abs/2402.05120) [[G-Scholar]](https://scholar.google.com/scholar?q=More%20Agents%20Is%20All%20You%20Need)
- `ReConcile` [Chen et al., ACL 2024] **Reconcile: Round-table conference improves reasoning via consensus among diverse LLMs** [[Paper]](https://aclanthology.org/2024.acl-long.381/) [[G-Scholar]](https://scholar.google.com/scholar?q=Reconcile%3A%20Round-table%20conference%20improves%20reasoning%20via%20consensus%20among%20diverse%20LLMs)
- `METAL` [Li et al., ACL 2025] **Metal: A multi-agent framework for chart generation with test-time scaling** [[Paper]](https://arxiv.org/abs/2502.17651) [[G-Scholar]](https://scholar.google.com/scholar?q=Metal%3A%20A%20multi-agent%20framework%20for%20chart%20generation%20with%20test-time%20scaling)
- `MoA` [Wang et al., ICLR 2025] **Mixture-of-agents enhances large language model capabilities** [[Paper]](https://arxiv.org/abs/2406.04692) [[G-Scholar]](https://scholar.google.com/scholar?q=Mixture-of-agents%20enhances%20large%20language%20model%20capabilities)
- `DiverseAgentEntropy` [Feng et al., Findings of EMNLP 2025] **Rethinking LLM Uncertainty: A Multi-Agent Approach to Estimating Black-Box Model Uncertainty** [[Paper]](https://aclanthology.org/2025.findings-emnlp.660/) [[G-Scholar]](https://scholar.google.com/scholar?q=Rethinking%20LLM%20Uncertainty%3A%20A%20Multi-Agent%20Approach%20to%20Estimating%20Black-Box%20Model%20Uncertainty)
- `MacNet` [Qian et al., ICLR 2025] **Scaling large language model-based multi-agent collaboration** [[Paper]](https://arxiv.org/abs/2406.07155) [[G-Scholar]](https://scholar.google.com/scholar?q=Scaling%20large%20language%20model-based%20multi-agent%20collaboration)
- `TUMIX` [Chen et al., arXiv 2025] **TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture** [[Paper]](https://arxiv.org/abs/2510.01279) [[G-Scholar]](https://scholar.google.com/scholar?q=TUMIX%3A%20Multi-Agent%20Test-Time%20Scaling%20with%20Tool-Use%20Mixture)

## 3. Theory and Evaluation of Test-Time Scaling

### 3.1 Sampling, Verification, and Scaling Laws

- [Nguyen et al., arXiv 2024] **When is the consistent prediction likely to be a correct prediction?** [[Paper]](https://arxiv.org/abs/2407.05778) [[G-Scholar]](https://scholar.google.com/scholar?q=When%20is%20the%20consistent%20prediction%20likely%20to%20be%20a%20correct%20prediction%3F)
- [Zhou et al., NeurIPS 2025] **A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning** [[Paper]](https://arxiv.org/abs/2510.15444) [[G-Scholar]](https://scholar.google.com/scholar?q=A%20Theoretical%20Study%20on%20Bridging%20Internal%20Probability%20and%20Self-Consistency%20for%20LLM%20Reasoning)
- [Huang et al., ICML 2025] **Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment** [[Paper]](https://arxiv.org/abs/2503.21878) [[G-Scholar]](https://scholar.google.com/scholar?q=Is%20Best-of-N%20the%20Best%20of%20Them%3F%20Coverage%2C%20Scaling%2C%20and%20Optimality%20in%20Inference-Time%20Alignment)
- [Setlur et al., ICML 2025] **Scaling Test-Time Compute Without Verification or RL is Suboptimal** [[Paper]](https://arxiv.org/abs/2502.12118) [[G-Scholar]](https://scholar.google.com/scholar?q=Scaling%20Test-Time%20Compute%20Without%20Verification%20or%20RL%20is%20Suboptimal)

### 3.2 Compute Allocation and Diminishing Returns

- [Wu et al., arXiv 2024] **A comparative study on reasoning patterns of openai's o1 model** [[Paper]](https://arxiv.org/abs/2410.13639) [[G-Scholar]](https://scholar.google.com/scholar?q=A%20comparative%20study%20on%20reasoning%20patterns%20of%20openai's%20o1%20model)
- [Chen et al., NeurIPS 2024] **Are More LLM Calls All You Need? Towards the Scaling Properties of Compound AI Systems** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/51173cf34c5faac9796a47dc2fdd3a71-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Are%20More%20LLM%20Calls%20All%20You%20Need%3F%20Towards%20the%20Scaling%20Properties%20of%20Compound%20AI%20Systems)
- `Tree-Search-Based Inference` [Chen et al., ACL 2024] **When is tree search useful for LLM planning? it depends on the discriminator** [[Paper]](https://aclanthology.org/2024.acl-long.738/) [[G-Scholar]](https://scholar.google.com/scholar?q=When%20is%20tree%20search%20useful%20for%20LLM%20planning%3F%20it%20depends%20on%20the%20discriminator)
- `Long CoT` [Yeo et al., ICLR Workshop 2025] **Demystifying long chain-of-thought reasoning in LLMs** [[Paper]](https://arxiv.org/abs/2502.03373) [[G-Scholar]](https://scholar.google.com/scholar?q=Demystifying%20long%20chain-of-thought%20reasoning%20in%20LLMs)
- [Li et al., arXiv 2026] **Benchmark Test-Time Scaling of General LLM Agents** [[Paper]](https://arxiv.org/abs/2602.18998) [[G-Scholar]](https://scholar.google.com/scholar?q=Benchmark%20Test-Time%20Scaling%20of%20General%20LLM%20Agents)
- [Wu et al., ICLR Workshop 2026] **When More is Less: Understanding Chain-of-Thought Length in LLMs** [[Paper]](https://arxiv.org/abs/2502.07266) [[G-Scholar]](https://scholar.google.com/scholar?q=When%20More%20is%20Less%3A%20Understanding%20Chain-of-Thought%20Length%20in%20LLMs)

### 3.3 Reliability, Safety, and Adversarial Evaluation

- [Raina et al., EMNLP 2024] **Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment** [[Paper]](https://arxiv.org/abs/2402.14016) [[G-Scholar]](https://scholar.google.com/scholar?q=Is%20LLM-as-a-Judge%20Robust%3F%20Investigating%20Universal%20Adversarial%20Attacks%20on%20Zero-shot%20LLM%20Assessment)
- [Stechly et al., ICLR 2025] **On the self-verification limitations of large language models on reasoning and planning tasks** [[Paper]](https://arxiv.org/abs/2402.08115) [[G-Scholar]](https://scholar.google.com/scholar?q=On%20the%20self-verification%20limitations%20of%20large%20language%20models%20on%20reasoning%20and%20planning%20tasks)
- [Nahin et al., ICML 2026] **Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models** [[Paper]](https://arxiv.org/abs/2510.08592) [[G-Scholar]](https://scholar.google.com/scholar?q=Less%20Diverse%2C%20Less%20Safe%3A%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Models)

### 3.4 Persistence and Amortization Boundaries

- `MiND` [Song et al., ICLR 2025] **Mind the gap: Examining the self-improvement capabilities of large language models** [[Paper]](https://arxiv.org/abs/2412.02674) [[G-Scholar]](https://scholar.google.com/scholar?q=Mind%20the%20gap%3A%20Examining%20the%20self-improvement%20capabilities%20of%20large%20language%20models)
- [Huang et al., ICLR 2025] **Self-improvement in language models: The sharpening mechanism** [[Paper]](https://openreview.net/forum?id=WJaUkwci9o) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-improvement%20in%20language%20models%3A%20The%20sharpening%20mechanism)
- [Lin et al., arXiv 2025] **Sleep-time compute: Beyond inference scaling at test-time** [[Paper]](https://arxiv.org/abs/2504.13171) [[G-Scholar]](https://scholar.google.com/scholar?q=Sleep-time%20compute%3A%20Beyond%20inference%20scaling%20at%20test-time)
