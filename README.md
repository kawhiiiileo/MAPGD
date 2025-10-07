
# MAPGD: Multi-Agent Prompt Gradient Descent for Collaborative Prompt Optimization

### 👨‍💻 Authors

Yichen Han¹, Yuhang Han², Bojun Liu³, Zhengpeng Zhou⁴, Guanyu Liu⁵, Zeng Zhang¹,  
Yang Yang⁶, Wenli Wang⁶, Isaac N Shi⁶, Yunyan Zhang⁶, Lewei He¹✉, Tianyu Shi⁷✉  
### 🏫 Affiliations
1. South China Normal University, Guangzhou, China  
2. Northwestern Polytechnical University, Xi'an, China  
3. University of Sydney, Sydney, Australia  
4. Shanghai Jiaotong University, Shanghai, China  
5. University of Macau, Macau, China  
6. Silicon Sapiens LLC, Jinan, China  
7. University of Toronto, Toronto, Canada  
---
### ✉️ Corresponding Authors
- **Lewei He** · [helewei@m.scnu.edu.cn](mailto:helewei@m.scnu.edu.cn)  
- **Tianyu Shi** · [tys@cs.toronto.edu](mailto:tys@cs.toronto.edu)  

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

