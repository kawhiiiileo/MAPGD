# MAPGD: Multi-Agent Prompt Gradient Descent

> Collaborative Prompt Optimization with Multi-Agent Gradient-like Feedback

---

## 🚀 Abstract
Prompt design critically affects the performance of large language models (LLMs). Existing optimization methods often rely on single-agent heuristics, which lack diversity, collaboration, and robustness.  
**MAPGD** introduces a multi-agent framework where each agent explores prompts from different perspectives, generates textual “gradients,” and collaboratively improves prompts via **beam search**, **semantic fusion**, and **bandit-based selection**.  
This approach improves diversity, semantic directionality, and interpretability—offering a scalable and effective solution for real-world prompt engineering.

---

## 🧩 Core Idea

- **Multiple Agents** → Explore diverse semantic directions  
- **Textual Gradients** → Semantic feedback analogous to numerical gradients  
- **Gradient Fusion** → Merge and align multiple prompt edits  
- **Beam Search** → Expand candidate prompt space efficiently  
- **Bandit Selection** → Identify best prompts with minimal evaluation cost  

---

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

@article{mapgd2025,
  title={MAPGD: Multi-Agent Prompt Gradient Descent for Collaborative Prompt Optimization},
  author={Your Name et al.},
  year={2025},
  journal={Preprint}
}
