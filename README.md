# MAPGD: Multi-Agent Prompt Gradient Descent for Collaborative Prompt Optimization
### 👨‍💻 Authors
- **Yichen Han** — South China Normal University, Guangzhou, China · [2024024502@m.scnu.edu.cn](mailto:2024024502@m.scnu.edu.cn)  
- **Yuhang Han** — Northwestern Polytechnical University, Xi'an, China · [hanyh@mail.nwpu.edu.cn](mailto:hanyh@mail.nwpu.edu.cn)  
- **Bojun Liu** — University of Sydney, Sydney, Australia · [liubojun9999@gmail.com](mailto:liubojun9999@gmail.com)  
- **Zhengpeng Zhou** — Shanghai Jiaotong University, Shanghai, China · [alex_chou@sjtu.edu.cn](mailto:alex_chou@sjtu.edu.cn)  
- **Guanyu Liu** — University of Macau, Macau, China · [dc32352@um.edu.mo](mailto:dc32352@um.edu.mo)  
- **Zeng Zhang** — South China Normal University, Guangzhou, China · [2024024588@m.scnu.edu.cn](mailto:2024024588@m.scnu.edu.cn)  
- **Yang Yang** — Silicon Sapiens LLC, Jinan, China · [yang@esapiens.ai](mailto:yang@esapiens.ai)  
- **Wenli Wang** — Silicon Sapiens LLC, Jinan, China · [wenli@esapiens.ai](mailto:wenli@esapiens.ai)  
- **Isaac N Shi** — Silicon Sapiens LLC, Jinan, China · [isaac@goldensection.com](mailto:isaac@goldensection.com)  
- **Yunyan Zhang** — Silicon Sapiens LLC, Jinan, China · [yunyan@fangyingmobile.com](mailto:yunyan@fangyingmobile.com)  
- **Lewei He** (✉️ Corresponding author) — South China Normal University, Guangzhou, China · [helewei@m.scnu.edu.cn](mailto:helewei@m.scnu.edu.cn)  
- **Tianyu Shi** (✉️ Corresponding author) — University of Toronto, Toronto, Canada · [tys@cs.toronto.edu](mailto:tys@cs.toronto.edu)  
---
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

