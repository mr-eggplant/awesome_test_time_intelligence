# Awesome Test-Time Intelligence Applications [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated paper list for application domains of Test-Time Intelligence (TTI), organized by deployment domain and task family.

[← Overview](README.md) · [All papers](ALL_PAPERS.md)

## Contents

- [1. Vision Perception](#1-vision-perception)
  - [1.1 Image Classification](#11-image-classification)
  - [1.2 Point-Cloud Classification](#12-point-cloud-classification)
  - [1.3 Segmentation](#13-segmentation)
  - [1.4 Detection and Tracking](#14-detection-and-tracking)
  - [1.5 Video Understanding](#15-video-understanding)
- [2. Generative Models](#2-generative-models)
  - [2.1 Image Restoration and Synthesis](#21-image-restoration-and-synthesis)
  - [2.2 Video Generation](#22-video-generation)
  - [2.3 3D Generation and Reconstruction](#23-3d-generation-and-reconstruction)
  - [2.4 Inference-Time Scaling for Generative Models](#24-inference-time-scaling-for-generative-models)
- [3. Language and Multimodal Models](#3-language-and-multimodal-models)
  - [3.1 Language Reasoning, Retrieval, and Adaptation](#31-language-reasoning-retrieval-and-adaptation)
  - [3.2 Vision-Language Adaptation and Verification](#32-vision-language-adaptation-and-verification)
  - [3.3 Spatial Reasoning](#33-spatial-reasoning)
  - [3.4 Speech and Audio Processing](#34-speech-and-audio-processing)
- [4. Embodied AI and Robotics](#4-embodied-ai-and-robotics)
  - [4.1 Policy Adaptation to Unseen Conditions](#41-policy-adaptation-to-unseen-conditions)
  - [4.2 Online Skill Improvement](#42-online-skill-improvement)
  - [4.3 Planning-Time Scaling](#43-planning-time-scaling)
- [5. Agentic AI](#5-agentic-ai)
  - [5.1 Deployment-Time Agent Adaptation and Learning](#51-deployment-time-agent-adaptation-and-learning)
  - [5.2 Recursive Agent Self-Improvement](#52-recursive-agent-self-improvement)
- [6. Healthcare and Personalized AI](#6-healthcare-and-personalized-ai)
  - [6.1 Clinical Environment Adaptation](#61-clinical-environment-adaptation)
  - [6.2 Patient-Specific Adaptation](#62-patient-specific-adaptation)
  - [6.3 Federated and Privacy-Constrained Personalization](#63-federated-and-privacy-constrained-personalization)

## 1. Vision Perception

### 1.1 Image Classification

- `CoTTA` [Wang et al., CVPR 2022] **Continual Test-Time Domain Adaptation** [[Paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Continual%20Test-Time%20Domain%20Adaptation) [[CODE]](https://github.com/qinenergy/cotta)
- `FOA` [Niu et al., ICML 2024] **Test-Time Model Adaptation with Only Forward Passes** [[Paper]](https://arxiv.org/abs/2404.01650) [[G-Scholar]](https://scholar.google.com/scholar?q=Test-Time%20Model%20Adaptation%20with%20Only%20Forward%20Passes) [[CODE]](https://github.com/mr-eggplant/FOA)
- `SAR²` [Niu et al., arXiv 2025] **Adapt in the wild: Test-time entropy minimization with sharpness and feature regularization** [[Paper]](https://arxiv.org/abs/2509.04977) [[G-Scholar]](https://scholar.google.com/scholar?q=Adapt%20in%20the%20wild%3A%20Test-time%20entropy%20minimization%20with%20sharpness%20and%20feature%20regularization)
- `EATA-C` [Tan et al., TPAMI 2025] **Uncertainty-calibrated test-time model adaptation without forgetting** [[Paper]](https://arxiv.org/abs/2403.11491) [[G-Scholar]](https://scholar.google.com/scholar?q=Uncertainty-calibrated%20test-time%20model%20adaptation%20without%20forgetting)
- `CMDA` [Zhang et al., IEEE TCSVT 2026] **Collaborative Model and Data Adaptation at Test Time** [[Paper]](https://doi.org/10.1109/TCSVT.2026.3661181) [[G-Scholar]](https://scholar.google.com/scholar?q=Collaborative%20Model%20and%20Data%20Adaptation%20at%20Test%20Time)

### 1.2 Point-Cloud Classification

- `BFTT3D` [Wang et al., CVPR 2024] **Backpropagation-free network for 3D test-time adaptation** [[Paper]](https://arxiv.org/abs/2403.18442) [[G-Scholar]](https://scholar.google.com/scholar?q=Backpropagation-free%20network%20for%203D%20test-time%20adaptation)
- `CloudFixer` [Shim et al., ECCV 2024] **CloudFixer: Test-time adaptation for 3D point clouds via diffusion-guided geometric transformation** [[Paper]](https://arxiv.org/abs/2407.16193) [[G-Scholar]](https://scholar.google.com/scholar?q=CloudFixer%3A%20Test-time%20adaptation%20for%203D%20point%20clouds%20via%20diffusion-guided%20geometric%20transformation)
- `GSDTTA` [Wei et al., ICCV 2025] **3D Test-time Adaptation via Graph Spectral Driven Point Shift** [[Paper]](https://arxiv.org/abs/2507.18225) [[G-Scholar]](https://scholar.google.com/scholar?q=3D%20Test-time%20Adaptation%20via%20Graph%20Spectral%20Driven%20Point%20Shift)

### 1.3 Segmentation

- `AuxAdapt` [Zhang et al., WACV 2022] **AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation** [[Paper]](https://openaccess.thecvf.com/content/WACV2022/papers/Zhang_AuxAdapt_Stable_and_Efficient_Test-Time_Adaptation_for_Temporally_Consistent_Video_WACV_2022_paper.pdf) [[G-Scholar]](https://scholar.google.com/scholar?q=AuxAdapt%3A%20Stable%20and%20Efficient%20Test-Time%20Adaptation%20for%20Temporally%20Consistent%20Video%20Semantic%20Segmentation)
- `DIGA` [Wang et al., CVPR 2023] **Dynamically instance-guided adaptation: A backward-free approach for test-time domain adaptive semantic segmentation** [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Dynamically_Instance-Guided_Adaptation_A_Backward-Free_Approach_for_Test-Time_Domain_Adaptive_CVPR_2023_paper.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Dynamically%20instance-guided%20adaptation%3A%20A%20backward-free%20approach%20for%20test-time%20domain%20adaptive%20semantic%20segmentation)
- `TeSLA` [Tomar et al., CVPR 2023] **Tesla: Test-time self-learning with automatic adversarial augmentation** [[Paper]](https://arxiv.org/abs/2303.09870) [[G-Scholar]](https://scholar.google.com/scholar?q=Tesla%3A%20Test-time%20self-learning%20with%20automatic%20adversarial%20augmentation)
- `APCoTTA` [Zhu et al., MICCAI 2023] **Uncertainty and shape-aware continual test-time adaptation for cross-domain segmentation of medical images** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-43898-1_63) [[G-Scholar]](https://scholar.google.com/scholar?q=Uncertainty%20and%20shape-aware%20continual%20test-time%20adaptation%20for%20cross-domain%20segmentation%20of%20medical%20images)
- `VPTTA` [Chen et al., CVPR 2024] **Each test image deserves a specific prompt: Continual test-time adaptation for 2d medical image segmentation** [[Paper]](https://arxiv.org/abs/2311.18363) [[G-Scholar]](https://scholar.google.com/scholar?q=Each%20test%20image%20deserves%20a%20specific%20prompt%3A%20Continual%20test-time%20adaptation%20for%202d%20medical%20image%20segmentation)
- `PromptCAL` [Lei et al., SMC 2025] **PromptCAL: Entropy-Calibrated and Prompt-Tuned Test-Time Adaptation for Semantic Segmentation** [[Paper]](https://tugraz.elsevierpure.com/en/publications/4785b25c-a579-40f0-9ac7-de5cf1b2b033/) [[G-Scholar]](https://scholar.google.com/scholar?q=PromptCAL%3A%20Entropy-Calibrated%20and%20Prompt-Tuned%20Test-Time%20Adaptation%20for%20Semantic%20Segmentation)

### 1.4 Detection and Tracking

- `DARTH` [Segu et al., ICCV 2023] **DARTH: Holistic Test-time Adaptation for Multiple Object Tracking** [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Segu_DARTH_Holistic_Test-time_Adaptation_for_Multiple_Object_Tracking_ICCV_2023_paper.html) [[G-Scholar]](https://scholar.google.com/scholar?q=DARTH%3A%20Holistic%20Test-time%20Adaptation%20for%20Multiple%20Object%20Tracking)
- `DPO` [Chen et al., ACM MM 2024] **DPO: Dual-perturbation optimization for test-time adaptation in 3D object detection** [[Paper]](https://arxiv.org/abs/2406.13891) [[G-Scholar]](https://scholar.google.com/scholar?q=DPO%3A%20Dual-perturbation%20optimization%20for%20test-time%20adaptation%20in%203D%20object%20detection)
- `MonoTTA` [Lin et al., ECCV 2024] **MonoTTA: Fully Test-Time Adaptation for Monocular 3D Object Detection** [[Paper]](https://arxiv.org/abs/2405.19682) [[G-Scholar]](https://scholar.google.com/scholar?q=MonoTTA%3A%20Fully%20Test-Time%20Adaptation%20for%20Monocular%203D%20Object%20Detection)

### 1.5 Video Understanding

- `ViTTA` [Lin et al., CVPR 2023] **Video test-time adaptation for action recognition** [[Paper]](https://arxiv.org/abs/2211.15393) [[G-Scholar]](https://scholar.google.com/scholar?q=Video%20test-time%20adaptation%20for%20action%20recognition)
- `MC-TTA` [Xiong et al., CVPR 2024] **Modality-collaborative test-time adaptation for action recognition** [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_Modality-Collaborative_Test-Time_Adaptation_for_Action_Recognition_CVPR_2024_paper.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Modality-collaborative%20test-time%20adaptation%20for%20action%20recognition)
- `ST2ST` [Fahim et al., CVPR Workshop 2024] **ST2ST: Self-Supervised Test-time Adaptation for Video Action Recognition** [[Paper]](https://tta-cvpr2024.github.io/_downloads/71c70b99c912cb3c70bd2f6cd13023b7/matp_5_st2st_self_supervised_test_tim.pdf) [[G-Scholar]](https://scholar.google.com/scholar?q=ST2ST%3A%20Self-Supervised%20Test-time%20Adaptation%20for%20Video%20Action%20Recognition)
- `T3AL` [Liberatori et al., CVPR 2024] **Test-time zero-shot temporal action localization** [[Paper]](https://arxiv.org/abs/2404.05426) [[G-Scholar]](https://scholar.google.com/scholar?q=Test-time%20zero-shot%20temporal%20action%20localization)
- `Skeleton-Cache` [Zhu et al., NeurIPS 2025] **Boosting Skeleton-based Zero-Shot Action Recognition with Training-Free Test-Time Adaptation** [[Paper]](https://arxiv.org/abs/2512.11458) [[G-Scholar]](https://scholar.google.com/scholar?q=Boosting%20Skeleton-based%20Zero-Shot%20Action%20Recognition%20with%20Training-Free%20Test-Time%20Adaptation)

## 2. Generative Models

### 2.1 Image Restoration and Synthesis

- [Shocher et al., CVPR 2018] **"Zero-Shot" Super-Resolution Using Deep Internal Learning** [[Paper]](https://ieeexplore.ieee.org/abstract/document/8578427/) [[G-Scholar]](https://scholar.google.com/scholar?q=%22Zero-Shot%22%20Super-Resolution%20Using%20Deep%20Internal%20Learning)
- `DIP` [Ulyanov et al., CVPR 2018] **Deep Image Prior** [[Paper]](https://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Deep%20Image%20Prior) [[CODE]](https://github.com/DmitryUlyanov/deep-image-prior)
- `SRTTA` [Deng et al., NeurIPS 2023] **Efficient test-time adaptation for super-resolution with second-order degradation and reconstruction** [[Paper]](https://arxiv.org/abs/2310.19011) [[G-Scholar]](https://scholar.google.com/scholar?q=Efficient%20test-time%20adaptation%20for%20super-resolution%20with%20second-order%20degradation%20and%20reconstruction)
- `ADPforAIR` [Shao et al., IEEE Transactions on Fuzzy Systems 2024] **Adaptive fuzzy degradation perception based on CLIP prior for all-in-one image restoration** [[Paper]](https://ieeexplore.ieee.org/abstract/document/10791873/) [[G-Scholar]](https://scholar.google.com/scholar?q=Adaptive%20fuzzy%20degradation%20perception%20based%20on%20CLIP%20prior%20for%20all-in-one%20image%20restoration)
- `Decorruptor-CM` [Oh et al., ECCV 2024] **Efficient diffusion-driven corruption editor for test-time adaptation** [[Paper]](https://arxiv.org/abs/2403.10911) [[G-Scholar]](https://scholar.google.com/scholar?q=Efficient%20diffusion-driven%20corruption%20editor%20for%20test-time%20adaptation)
- `TTT-MIM` [Mansour et al., ECCV 2024] **TTT-MIM: test-time training with masked image modeling for denoising distribution shifts** [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1921_ECCV_2024_paper.php) [[G-Scholar]](https://scholar.google.com/scholar?q=TTT-MIM%3A%20test-time%20training%20with%20masked%20image%20modeling%20for%20denoising%20distribution%20shifts)
- `CauSiam` [Cui et al., IJCV 2025] **Continual test-time adaptation for single image defocus deblurring via causal Siamese networks** [[Paper]](https://arxiv.org/abs/2501.09052) [[G-Scholar]](https://scholar.google.com/scholar?q=Continual%20test-time%20adaptation%20for%20single%20image%20defocus%20deblurring%20via%20causal%20Siamese%20networks)

### 2.2 Video Generation

- `SETA` [Chen et al., arXiv 2023] **Open-world pose transfer via sequential test-time adaption** [[Paper]](https://arxiv.org/abs/2303.10945) [[G-Scholar]](https://scholar.google.com/scholar?q=Open-world%20pose%20transfer%20via%20sequential%20test-time%20adaption)
- `CustomTTT` [Bi et al., AAAI 2025] **Customttt: Motion and appearance customized video generation via test-time training** [[Paper]](https://arxiv.org/abs/2412.15646) [[G-Scholar]](https://scholar.google.com/scholar?q=Customttt%3A%20Motion%20and%20appearance%20customized%20video%20generation%20via%20test-time%20training)
- `TTOM` [Qu et al., arXiv 2025] **TTOM: Test-Time Optimization and Memorization for Compositional Video Generation** [[Paper]](https://arxiv.org/abs/2510.07940) [[G-Scholar]](https://scholar.google.com/scholar?q=TTOM%3A%20Test-Time%20Optimization%20and%20Memorization%20for%20Compositional%20Video%20Generation)
- `TTC` [Xiang et al., arXiv 2026] **Pathwise Test-Time Correction for Autoregressive Long Video Generation** [[Paper]](https://arxiv.org/abs/2602.05871) [[G-Scholar]](https://scholar.google.com/scholar?q=Pathwise%20Test-Time%20Correction%20for%20Autoregressive%20Long%20Video%20Generation)

### 2.3 3D Generation and Reconstruction

- `Surrogate Adaptation` [Lumentut et al., ACM MM 2022] **3D body reconstruction revisited: Exploring the test-time 3D body mesh refinement strategy via surrogate adaptation** [[Paper]](https://doi.org/10.1145/3503161.3547842) [[G-Scholar]](https://scholar.google.com/scholar?q=3D%20body%20reconstruction%20revisited%3A%20Exploring%20the%20test-time%203D%20body%20mesh%20refinement%20strategy%20via%20surrogate%20adaptation)
- `3DHR-Co` [Lumentut et al., IEEE Access 2023] **3DHR-Co: A Collaborative Test-Time Refinement Framework for In-the-Wild 3D Human-Body Reconstruction Task** [[Paper]](https://arxiv.org/abs/2310.01291) [[G-Scholar]](https://scholar.google.com/scholar?q=3DHR-Co%3A%20A%20Collaborative%20Test-Time%20Refinement%20Framework%20for%20In-the-Wild%203D%20Human-Body%20Reconstruction%20Task)
- `DreamFusion` [Poole et al., ICLR 2023] **DreamFusion: Text-to-3D Using 2D Diffusion** [[Paper]](https://arxiv.org/abs/2209.14988) [[G-Scholar]](https://scholar.google.com/scholar?q=DreamFusion%3A%20Text-to-3D%20Using%202D%20Diffusion)
- `Shape Completion TTT` [Schopf-Kuester et al., ICML Workshop 2024] **3D Shape Completion with Test-Time Training** [[Paper]](https://openreview.net/forum?id=E7B97EiFYl) [[G-Scholar]](https://scholar.google.com/scholar?q=3D%20Shape%20Completion%20with%20Test-Time%20Training)
- `PointMAC` [Jiang et al., NeurIPS 2025] **PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion** [[Paper]](https://arxiv.org/abs/2510.10365) [[G-Scholar]](https://scholar.google.com/scholar?q=PointMAC%3A%20Meta-Learned%20Adaptation%20for%20Robust%20Test-Time%20Point%20Cloud%20Completion)
- `TTT3R` [Chen et al., ICLR 2026] **TTT3R: 3D Reconstruction as Test-Time Training** [[Paper]](https://arxiv.org/abs/2509.26645) [[G-Scholar]](https://scholar.google.com/scholar?q=TTT3R%3A%203D%20Reconstruction%20as%20Test-Time%20Training)

### 2.4 Inference-Time Scaling for Generative Models

- `InitNO` [Guo et al., CVPR 2024] **InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization** [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_InitNO_Boosting_Text-to-Image_Diffusion_Models_via_Initial_Noise_Optimization_CVPR_2024_paper.html) [[G-Scholar]](https://scholar.google.com/scholar?q=InitNO%3A%20Boosting%20Text-to-Image%20Diffusion%20Models%20via%20Initial%20Noise%20Optimization) [[CODE]](https://github.com/xiefan-guo/initno)
- `FK steering` [Singhal et al., ICML 2025] **A General Framework for Inference-time Scaling and Steering of Diffusion Models** [[Paper]](https://arxiv.org/abs/2501.06848) [[G-Scholar]](https://scholar.google.com/scholar?q=A%20General%20Framework%20for%20Inference-time%20Scaling%20and%20Steering%20of%20Diffusion%20Models)
- [Ma et al., arXiv 2025] **Inference-time scaling for diffusion models beyond scaling denoising steps** [[Paper]](https://arxiv.org/abs/2501.09732) [[G-Scholar]](https://scholar.google.com/scholar?q=Inference-time%20scaling%20for%20diffusion%20models%20beyond%20scaling%20denoising%20steps)
- [Zhang et al., NeurIPS Workshop 2025] **Inference-time Scaling of Diffusion Models through Classical Search** [[Paper]](https://neurips.cc/virtual/2025/125430) [[G-Scholar]](https://scholar.google.com/scholar?q=Inference-time%20Scaling%20of%20Diffusion%20Models%20through%20Classical%20Search)
- `Reflect-DiT` [Li et al., ICCV 2025] **Reflect-dit: Inference-time scaling for text-to-image diffusion transformers via in-context reflection** [[Paper]](https://arxiv.org/abs/2503.12271) [[G-Scholar]](https://scholar.google.com/scholar?q=Reflect-dit%3A%20Inference-time%20scaling%20for%20text-to-image%20diffusion%20transformers%20via%20in-context%20reflection)
- `SANA` [Xie et al., ICML 2025] **SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer** [[Paper]](https://arxiv.org/abs/2501.18427) [[G-Scholar]](https://scholar.google.com/scholar?q=SANA%201.5%3A%20Efficient%20Scaling%20of%20Training-Time%20and%20Inference-Time%20Compute%20in%20Linear%20Diffusion%20Transformer)
- `EvoSearch` [He et al., arXiv 2025] **Scaling image and video generation via test-time evolutionary search** [[Paper]](https://arxiv.org/abs/2505.17618) [[G-Scholar]](https://scholar.google.com/scholar?q=Scaling%20image%20and%20video%20generation%20via%20test-time%20evolutionary%20search)
- [Ramesh et al., NeurIPS 2025] **Test-time scaling of diffusion models via noise trajectory search** [[Paper]](https://papers.nips.cc/paper_files/paper/2025/hash/7e62167d35d1e830fd6afe5c899ed124-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Test-time%20scaling%20of%20diffusion%20models%20via%20noise%20trajectory%20search)
- [Kim et al., ICML 2026] **Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models** [[Paper]](https://arxiv.org/abs/2602.03211) [[G-Scholar]](https://scholar.google.com/scholar?q=Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models)
- `ScalingAR` [Chen et al., arXiv 2026] **ScalingAR: Scaling Confidence for Autoregressive Image Generation** [[Paper]](https://arxiv.org/abs/2509.26376) [[G-Scholar]](https://scholar.google.com/scholar?q=ScalingAR%3A%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation)

## 3. Language and Multimodal Models

### 3.1 Language Reasoning, Retrieval, and Adaptation

- [Cobbe et al., arXiv 2021] **Training Verifiers to Solve Math Word Problems** [[Paper]](https://arxiv.org/abs/2110.14168) [[G-Scholar]](https://scholar.google.com/scholar?q=Training%20Verifiers%20to%20Solve%20Math%20Word%20Problems)
- `Self-Consistency` [Wang et al., ICLR 2023] **Self-Consistency Improves Chain of Thought Reasoning in Language Models** [[Paper]](https://arxiv.org/abs/2203.11171) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-Consistency%20Improves%20Chain%20of%20Thought%20Reasoning%20in%20Language%20Models)
- `ToT` [Yao et al., NeurIPS 2023] **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** [[Paper]](https://arxiv.org/abs/2305.10601) [[G-Scholar]](https://scholar.google.com/scholar?q=Tree%20of%20Thoughts%3A%20Deliberate%20Problem%20Solving%20with%20Large%20Language%20Models)
- `Self-RAG` [Asai et al., ICLR 2024] **Self-RAG: Learning to retrieve, generate, and critique through self-reflection** [[Paper]](https://openreview.net/forum?id=hSyW5go0v8) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-RAG%3A%20Learning%20to%20retrieve%2C%20generate%2C%20and%20critique%20through%20self-reflection)
- `TLM` [Hu et al., ICML 2025] **Test-Time Learning for Large Language Models** [[Paper]](https://arxiv.org/abs/2505.20633) [[G-Scholar]](https://scholar.google.com/scholar?q=Test-Time%20Learning%20for%20Large%20Language%20Models)

### 3.2 Vision-Language Adaptation and Verification

- `TPT` [Shu et al., NeurIPS 2022] **Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models** [[Paper]](https://openreview.net/forum?id=e8PVEkSa4Fq) [[G-Scholar]](https://scholar.google.com/scholar?q=Test-Time%20Prompt%20Tuning%20for%20Zero-Shot%20Generalization%20in%20Vision-Language%20Models) [[CODE]](https://github.com/azshue/TPT)
- `DN` [Zhou et al., NeurIPS 2023] **Test-Time Distribution Normalization for Contrastively Learned Visual-language Models** [[Paper]](https://papers.nips.cc/paper_files/paper/2023/hash/931db0b5a61f9db6c97c7e4bf068147d-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Test-Time%20Distribution%20Normalization%20for%20Contrastively%20Learned%20Visual-language%20Models)
- `RA-TTA` [Lee et al., ICLR 2025] **RA-TTA: Retrieval-Augmented Test-Time Adaptation for Vision-Language Models** [[Paper]](https://openreview.net/forum?id=V3zobHnS61) [[G-Scholar]](https://scholar.google.com/scholar?q=RA-TTA%3A%20Retrieval-Augmented%20Test-Time%20Adaptation%20for%20Vision-Language%20Models)
- `TTRV` [Singh et al., arXiv 2025] **TTRV: Test-Time Reinforcement Learning for Vision Language Models** [[Paper]](https://arxiv.org/abs/2510.06783) [[G-Scholar]](https://scholar.google.com/scholar?q=TTRV%3A%20Test-Time%20Reinforcement%20Learning%20for%20Vision%20Language%20Models)
- `VisualPRM` [Wang et al., arXiv 2025] **VisualPRM: An effective process reward model for multimodal reasoning** [[Paper]](https://arxiv.org/abs/2503.10291) [[G-Scholar]](https://scholar.google.com/scholar?q=VisualPRM%3A%20An%20effective%20process%20reward%20model%20for%20multimodal%20reasoning)

### 3.3 Spatial Reasoning

- `MindJourney` [Yang et al., NeurIPS 2025] **MindJourney: Test-Time Scaling with World Models for Spatial Reasoning** [[Paper]](https://arxiv.org/abs/2507.12508) [[G-Scholar]](https://scholar.google.com/scholar?q=MindJourney%3A%20Test-Time%20Scaling%20with%20World%20Models%20for%20Spatial%20Reasoning)
- `TangramSR` [Zong et al., arXiv 2026] **TangramSR: Can Vision-Language Models Reason in Continuous Geometric Space?** [[Paper]](https://arxiv.org/abs/2602.05570) [[G-Scholar]](https://scholar.google.com/scholar?q=TangramSR%3A%20Can%20Vision-Language%20Models%20Reason%20in%20Continuous%20Geometric%20Space%3F)
- `AVIC` [Yu et al., arXiv 2026] **When and how much to imagine: Adaptive test-time scaling with world models for visual spatial reasoning** [[Paper]](https://arxiv.org/abs/2602.08236) [[G-Scholar]](https://scholar.google.com/scholar?q=When%20and%20how%20much%20to%20imagine%3A%20Adaptive%20test-time%20scaling%20with%20world%20models%20for%20visual%20spatial%20reasoning)

### 3.4 Speech and Audio Processing

- `SUTA` [Lin et al., INTERSPEECH 2022] **Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition** [[Paper]](https://arxiv.org/abs/2203.14222) [[G-Scholar]](https://scholar.google.com/scholar?q=Listen%2C%20Adapt%2C%20Better%20WER%3A%20Source-free%20Single-utterance%20Test-time%20Adaptation%20for%20Automatic%20Speech%20Recognition)
- `AWMC` [Lee et al., ASRU 2023] **Awmc: Online test-time adaptation without mode collapse for continual adaptation** [[Paper]](https://ieeexplore.ieee.org/abstract/document/10389640) [[G-Scholar]](https://scholar.google.com/scholar?q=Awmc%3A%20Online%20test-time%20adaptation%20without%20mode%20collapse%20for%20continual%20adaptation)
- `SGEM` [Kim et al., INTERSPEECH 2023] **SGEM: Test-Time Adaptation for Automatic Speech Recognition via Sequential-Level Generalized Entropy Minimization** [[Paper]](https://arxiv.org/abs/2306.01981) [[G-Scholar]](https://scholar.google.com/scholar?q=SGEM%3A%20Test-Time%20Adaptation%20for%20Automatic%20Speech%20Recognition%20via%20Sequential-Level%20Generalized%20Entropy%20Minimization)
- `CEA` [Liu et al., EMNLP 2024] **Advancing test-time adaptation in wild acoustic test settings** [[Paper]](https://aclanthology.org/2024.emnlp-main.405/) [[G-Scholar]](https://scholar.google.com/scholar?q=Advancing%20test-time%20adaptation%20in%20wild%20acoustic%20test%20settings)
- `CSUTA/DSUTA` [Lin et al., EMNLP 2024] **Continual test-time adaptation for end-to-end speech recognition on noisy speech** [[Paper]](https://arxiv.org/abs/2406.11064) [[G-Scholar]](https://scholar.google.com/scholar?q=Continual%20test-time%20adaptation%20for%20end-to-end%20speech%20recognition%20on%20noisy%20speech)
- `E-BATS` [Dong et al., NeurIPS 2025] **E-BATS: Efficient backpropagation-free test-time adaptation for speech foundation models** [[Paper]](https://papers.nips.cc/paper_files/paper/2025/hash/ce6326ac4794bb04d5eb16f597446baf-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=E-BATS%3A%20Efficient%20backpropagation-free%20test-time%20adaptation%20for%20speech%20foundation%20models)

## 4. Embodied AI and Robotics

### 4.1 Policy Adaptation to Unseen Conditions

- `PAD` [Hansen et al., ICLR 2021] **Self-supervised policy adaptation during deployment** [[Paper]](https://openreview.net/forum?id=o_V-MjyyGV_) [[G-Scholar]](https://scholar.google.com/scholar?q=Self-supervised%20policy%20adaptation%20during%20deployment)
- `T4P` [Park et al., CVPR 2024] **T4p: Test-time training of trajectory prediction via masked autoencoder and actor-specific token memory** [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Park_T4P_Test-Time_Training_of_Trajectory_Prediction_via_Masked_Autoencoder_and_CVPR_2024_paper.html) [[G-Scholar]](https://scholar.google.com/scholar?q=T4p%3A%20Test-time%20training%20of%20trajectory%20prediction%20via%20masked%20autoencoder%20and%20actor-specific%20token%20memory)
- `Centaur` [Sima et al., arXiv 2025] **Centaur: Robust End-to-End Autonomous Driving with Test-Time Training** [[Paper]](https://arxiv.org/abs/2503.11650) [[G-Scholar]](https://scholar.google.com/scholar?q=Centaur%3A%20Robust%20End-to-End%20Autonomous%20Driving%20with%20Test-Time%20Training)
- `AdaptFly` [Chen et al., TCCN 2026] **AdaptFly: Prompt-Guided Adaptation of Foundation Models for Low-Altitude UAV Networks** [[Paper]](https://doi.org/10.1109/TCCN.2026.3658758) [[G-Scholar]](https://scholar.google.com/scholar?q=AdaptFly%3A%20Prompt-Guided%20Adaptation%20of%20Foundation%20Models%20for%20Low-Altitude%20UAV%20Networks)
- `BRIC` [Lim et al., AAAI 2026] **BRIC: Bridging Kinematic Plans and Physical Control at Test Time** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/39522) [[G-Scholar]](https://scholar.google.com/scholar?q=BRIC%3A%20Bridging%20Kinematic%20Plans%20and%20Physical%20Control%20at%20Test%20Time)
- `TT-VLA` [Liu et al., arXiv 2026] **On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning** [[Paper]](https://arxiv.org/abs/2601.06748) [[G-Scholar]](https://scholar.google.com/scholar?q=On-the-Fly%20VLA%20Adaptation%20via%20Test-Time%20Reinforcement%20Learning)

### 4.2 Online Skill Improvement

- [Mendez-Mendez et al., CoRL 2023] **Embodied Lifelong Learning for Task and Motion Planning** [[Paper]](https://proceedings.mlr.press/v229/mendez-mendez23a.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Embodied%20Lifelong%20Learning%20for%20Task%20and%20Motion%20Planning)
- [Tziafas et al., arXiv 2024] **Lifelong Robot Library Learning: Bootstrapping Composable and Generalizable Skills for Embodied Control with Language Models** [[Paper]](https://arxiv.org/abs/2406.18746) [[G-Scholar]](https://scholar.google.com/scholar?q=Lifelong%20Robot%20Library%20Learning%3A%20Bootstrapping%20Composable%20and%20Generalizable%20Skills%20for%20Embodied%20Control%20with%20Language%20Models)
- `Voyager` [Wang et al., TMLR 2024] **Voyager: An Open-Ended Embodied Agent with Large Language Models** [[Paper]](https://arxiv.org/abs/2305.16291) [[G-Scholar]](https://scholar.google.com/scholar?q=Voyager%3A%20An%20Open-Ended%20Embodied%20Agent%20with%20Large%20Language%20Models) [[CODE]](https://github.com/MineDojo/Voyager)
- `ReflectivePlanning` [Hong et al., arXiv 2026] **Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs** [[Paper]](https://arxiv.org/abs/2602.21198) [[G-Scholar]](https://scholar.google.com/scholar?q=Learning%20from%20Trials%20and%20Errors%3A%20Reflective%20Test-Time%20Planning%20for%20Embodied%20LLMs)

### 4.3 Planning-Time Scaling

- `DA-SIP` [Chun et al., NeurIPS 2025] **Dynamic Test-Time Compute Scaling in Control Policy: Difficulty-Aware Stochastic Interpolant Policy** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/49eadcc4a329fc6b74b9f8a82b78cbc3-Abstract-Conference.html) [[G-Scholar]](https://scholar.google.com/scholar?q=Dynamic%20Test-Time%20Compute%20Scaling%20in%20Control%20Policy%3A%20Difficulty-Aware%20Stochastic%20Interpolant%20Policy)
- `RoboMonkey` [Kwok et al., CoRL 2025] **RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models** [[Paper]](https://arxiv.org/abs/2506.17811) [[G-Scholar]](https://scholar.google.com/scholar?q=RoboMonkey%3A%20Scaling%20Test-Time%20Sampling%20and%20Verification%20for%20Vision-Language-Action%20Models) [[CODE]](https://github.com/robomonkey-vla/RoboMonkey)
- `RoVer` [Dai et al., arXiv 2025] **RoVer: Robot Reward Model as Test-Time Verifier for Vision-Language-Action Model** [[Paper]](https://arxiv.org/abs/2510.10975) [[G-Scholar]](https://scholar.google.com/scholar?q=RoVer%3A%20Robot%20Reward%20Model%20as%20Test-Time%20Verifier%20for%20Vision-Language-Action%20Model)
- `Steering VLA` [Yang et al., arXiv 2025] **Steering Vision-Language-Action Models as Anti-Exploration: A Test-Time Scaling Approach** [[Paper]](https://arxiv.org/abs/2512.02834) [[G-Scholar]](https://scholar.google.com/scholar?q=Steering%20Vision-Language-Action%20Models%20as%20Anti-Exploration%3A%20A%20Test-Time%20Scaling%20Approach)
- `CoVer-VLA` [Kwok et al., arXiv 2026] **Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment** [[Paper]](https://arxiv.org/abs/2602.12281) [[G-Scholar]](https://scholar.google.com/scholar?q=Scaling%20Verification%20Can%20Be%20More%20Effective%20than%20Scaling%20Policy%20Learning%20for%20Vision-Language-Action%20Alignment)
- `Verifier-Free TTS` [Jang et al., ICLR 2026] **Verifier-free Test-Time Sampling for Vision Language Action Models** [[Paper]](https://arxiv.org/abs/2510.05681) [[G-Scholar]](https://scholar.google.com/scholar?q=Verifier-free%20Test-Time%20Sampling%20for%20Vision%20Language%20Action%20Models)

## 5. Agentic AI

### 5.1 Deployment-Time Agent Adaptation and Learning

- `GTTA` [Chen et al., arXiv 2025] **Test-Time Adaptation for LLM Agents via Environment Interaction** [[Paper]](https://arxiv.org/abs/2511.04847) [[G-Scholar]](https://scholar.google.com/scholar?q=Test-Time%20Adaptation%20for%20LLM%20Agents%20via%20Environment%20Interaction)
- `MAS-on-the-Fly` [Liu et al., arXiv 2026] **MAS-on-the-Fly: Dynamic Adaptation of LLM-based Multi-Agent Systems at Test Time** [[Paper]](https://arxiv.org/abs/2602.13671) [[G-Scholar]](https://scholar.google.com/scholar?q=MAS-on-the-Fly%3A%20Dynamic%20Adaptation%20of%20LLM-based%20Multi-Agent%20Systems%20at%20Test%20Time)
- `TT-SI` [Acikgoz et al., Findings of ACL 2026] **TT-SI: Self-Improving LLM Agents with Test-Time Training** [[Paper]](https://aclanthology.org/2026.findings-acl.462/) [[G-Scholar]](https://scholar.google.com/scholar?q=TT-SI%3A%20Self-Improving%20LLM%20Agents%20with%20Test-Time%20Training)

### 5.2 Recursive Agent Self-Improvement

- [Robeyns et al., arXiv 2025] **A Self-Improving Coding Agent** [[Paper]](https://arxiv.org/abs/2504.15228) [[G-Scholar]](https://scholar.google.com/scholar?q=A%20Self-Improving%20Coding%20Agent)
- `Gödel Agent` [Yin et al., ACL 2025] **Gödel Agent: A Self-Referential Agent Framework for Recursively Self-Improvement** [[Paper]](https://aclanthology.org/2025.acl-long.1354/) [[G-Scholar]](https://scholar.google.com/scholar?q=G%C3%B6del%20Agent%3A%20A%20Self-Referential%20Agent%20Framework%20for%20Recursively%20Self-Improvement)
- `DGM` [Zhang et al., ICLR 2026] **Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents** [[Paper]](https://arxiv.org/abs/2505.22954) [[G-Scholar]](https://scholar.google.com/scholar?q=Darwin%20Godel%20Machine%3A%20Open-Ended%20Evolution%20of%20Self-Improving%20Agents)
- `Hyperagents` [Zhang et al., arXiv 2026] **Hyperagents** [[Paper]](https://arxiv.org/abs/2603.19461) [[G-Scholar]](https://scholar.google.com/scholar?q=Hyperagents)
- `MetaSkill-Evolve` [Wang et al., arXiv 2026] **MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution** [[Paper]](https://arxiv.org/abs/2607.05297) [[G-Scholar]](https://scholar.google.com/scholar?q=MetaSkill-Evolve%3A%20Recursive%20Self-Improvement%20of%20LLM%20Agents%20via%20Two-Timescale%20Meta-Skill%20Evolution)

## 6. Healthcare and Personalized AI

### 6.1 Clinical Environment Adaptation

- `TTA-DAE` [Karani et al., Medical Image Analysis 2021] **Test-time adaptable neural networks for robust medical image segmentation** [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841520302711) [[G-Scholar]](https://scholar.google.com/scholar?q=Test-time%20adaptable%20neural%20networks%20for%20robust%20medical%20image%20segmentation)
- `CertainTTA` [Dong et al., Information Fusion 2025] **CertainTTA: Estimating uncertainty for test-time adaptation on medical image segmentation** [[Paper]](https://www.sciencedirect.com/science/article/pii/S1566253525003732) [[G-Scholar]](https://scholar.google.com/scholar?q=CertainTTA%3A%20Estimating%20uncertainty%20for%20test-time%20adaptation%20on%20medical%20image%20segmentation)
- [Zhao et al., Pattern Recognition 2026] **Active test-time adaptation for continual medical image classification** [[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320326002530) [[G-Scholar]](https://scholar.google.com/scholar?q=Active%20test-time%20adaptation%20for%20continual%20medical%20image%20classification)

### 6.2 Patient-Specific Adaptation

- `Bi-TTA` [Li et al., ECCV 2024] **Bi-tta: Bidirectional test-time adapter for remote physiological measurement** [[Paper]](https://arxiv.org/abs/2409.17316) [[G-Scholar]](https://scholar.google.com/scholar?q=Bi-tta%3A%20Bidirectional%20test-time%20adapter%20for%20remote%20physiological%20measurement)
- [Wang et al., arXiv 2025] **A Robust Multi-Scale Framework with Test-Time Adaptation for sEEG-Based Speech Decoding** [[Paper]](https://arxiv.org/abs/2509.24700) [[G-Scholar]](https://scholar.google.com/scholar?q=A%20Robust%20Multi-Scale%20Framework%20with%20Test-Time%20Adaptation%20for%20sEEG-Based%20Speech%20Decoding) [[CODE]](https://github.com/lyyi599/MDM-TENT)
- [Jang et al., arXiv 2025] **Calibration-Free EEG-based Driver Drowsiness Detection with Online Test-Time Adaptation** [[Paper]](https://arxiv.org/abs/2511.22030) [[G-Scholar]](https://scholar.google.com/scholar?q=Calibration-Free%20EEG-based%20Driver%20Drowsiness%20Detection%20with%20Online%20Test-Time%20Adaptation)
- `NoMAD` [Karpowicz et al., Nature Communications 2025] **Stabilizing brain-computer interfaces through alignment of latent dynamics** [[Paper]](https://www.nature.com/articles/s41467-025-59652-y) [[G-Scholar]](https://scholar.google.com/scholar?q=Stabilizing%20brain-computer%20interfaces%20through%20alignment%20of%20latent%20dynamics) [[CODE]](https://github.com/snel-repo/nomad)

### 6.3 Federated and Privacy-Constrained Personalization

- `ATP` [Bao et al., NeurIPS 2023] **Adaptive test-time personalization for federated learning** [[Paper]](https://arxiv.org/abs/2310.18816) [[G-Scholar]](https://scholar.google.com/scholar?q=Adaptive%20test-time%20personalization%20for%20federated%20learning)
- `MSAFed` [Jin et al., JBHI 2025] **MSAFed: generalized multi-stage and adaptive federated learning for test-time medical segmentation** [[Paper]](https://doi.org/10.1109/JBHI.2025.3610103) [[G-Scholar]](https://scholar.google.com/scholar?q=MSAFed%3A%20generalized%20multi-stage%20and%20adaptive%20federated%20learning%20for%20test-time%20medical%20segmentation)
