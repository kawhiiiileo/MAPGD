<div align="center">

<h1> MAPGD: Multi-Agent Prompt Gradient Descent for Collaborative Prompt Optimization </h1>

<h4 align="center"> 

Yichen Han<sup>1</sup>,
Yuhang Han<sup>2</sup>,
Bojun Liu<sup>3</sup>,
Zhengpeng Zhou<sup>4</sup>,
Guanyu Liu<sup>5</sup>,\
Zeng Zhang<sup>1</sup>,
Yang Yang<sup>6</sup>,
Wenli Wang<sup>6</sup>,
Isaac N Shi<sup>6</sup>,
Yunyan Zhang<sup>6</sup>,\
Lewei He<sup>1✉</sup>,
Tianyu Shi<sup>7✉</sup>

<sup>1</sup>South China Normal University, Guangzhou, China  
<sup>2</sup>Northwestern Polytechnical University, Xi'an, China  
<sup>3</sup>University of Sydney, Sydney, Australia  
<sup>4</sup>Shanghai Jiaotong University, Shanghai, China  
<sup>5</sup>University of Macau, Macau, China  
<sup>6</sup>Silicon Sapiens LLC, Jinan, China  
<sup>7</sup>University of Toronto, Toronto, Canada  

<p>
<a href='https://arxiv.org/pdf/xxxx.xxxxx'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://github.com/xxx/MAPGD'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
</p>

</h4>
</div>

<p align='center'>
<img width="600" alt="image" src='./architecture.png'>
</p>

## 🚀 Abstract
Prompt design critically affects the performance of large language models (LLMs). Existing optimization methods often rely on single-agent heuristics, which lack diversity, collaboration, and robustness.  
**MAPGD** introduces a multi-agent framework where each agent explores prompts from different perspectives, generates textual “gradients,” and collaboratively improves prompts via **beam search**, **semantic fusion**, and **bandit-based selection**.  
This approach improves diversity, semantic directionality, and interpretability—offering a scalable and effective solution for real-world prompt engineering.
## 🧩 Core Idea

- **Multiple Agents** → Explore diverse semantic directions  
- **Textual Gradients** → Semantic feedback analogous to numerical gradients  
- **Gradient Fusion** → Merge and align multiple prompt edits  
- **Beam Search** → Expand candidate prompt space efficiently  
- **Bandit Selection** → Identify best prompts with minimal evaluation cost  
## ⚙️ System Workflow

```text
Input: Initial prompt p0, datasets D_train / D_dev

Iterative Optimization:
  1. Agents propose textual gradients & edits
  2. Gradients are clustered & fused
  3. Candidate prompts expanded (beam + paraphrasing)
  4. Bandit-based selection chooses top prompts
  5. Agents sync with best candidate

Output: Optimized prompt

