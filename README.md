# 🖥️ Multimodal Desktop Agent (OSWorld)

This project implements a **multimodal desktop agent** designed to operate **on top of the OSWorld benchmark**.  
The agent observes the GUI via **screenshots and accessibility trees**, reasons using a vision-language model, and performs actions by generating **PyAutoGUI code**.

⚠️ **Important:**  
This agent **cannot run standalone**. It must be executed within **OSWorld**, which provides the GUI environment, accessibility tree, and action execution layer.

---

## 📌 High-Level Idea

At each step, the agent:
1. Observes the current GUI (image + accessibility tree)
2. Maintains a compact summary of past actions
3. Uses an LLM to decide the next interaction
4. Outputs executable PyAutoGUI code

This design keeps the agent **context-aware** while remaining **token-efficient**.

---

## 🧠 Core Architecture

### Observation
- **Screenshot**: Raw GUI image encoded as base64
- **Accessibility Tree**: OS-provided UI tree, filtered and linearized into a compact table format

The accessibility tree is trimmed to fit a token budget, allowing the agent to reason about UI structure without exceeding context limits.

---

### Reasoning & Memory
- The agent maintains a **step-by-step summary** of previous actions.
- Each LLM response can include a structured `SUMMARY` block.
- Only summaries (not full history) are reused, reducing token usage while preserving progress awareness.

---

### Action Generation
- Actions are produced as **Python code blocks**.
- The code is executed by OSWorld using **PyAutoGUI**.
- Special terminal states:
  - `DONE` – task completed
  - `FAIL` – task failed
  - `WAIT` – pause / no-op

---

## 🧩 Code Structure

### `MyAgent`
Main agent class responsible for:
- Building multimodal prompts (text + image + accessibility tree)
- Tracking observations, actions, and summaries
- Parsing model outputs into executable actions
- Managing retries and error handling for LLM calls

Key methods:
- `predict()` – core decision loop
- `parse_actions()` – extracts PyAutoGUI code
- `parse_summary()` – extracts and numbers step summaries
- `reset()` – clears agent state between episodes

---

### Accessibility Utilities
- `linearize_accessibility_tree()`  
  Converts the XML accessibility tree into a tabular text format.
- `trim_accessibility_tree()`  
  Ensures the tree stays within a token budget.

These utilities make UI structure understandable to the language model.

---

### LLM Interface
- Uses `gpt-4o-mini` via OpenAI-compatible API
- Supports retry, rate-limit handling, and context-length recovery
- Multimodal prompt includes:
  - Task instruction
  - Previous summaries
  - Accessibility tree
  - Current screenshot

---

## 📊 Performance

- Evaluated on **complex OSWorld tasks**
- Domains: Ubuntu OS, Chrome, LibreOffice Calc
- **~10% success rate**, reflecting the difficulty of long-horizon desktop tasks

---

## ⚙️ Requirements & Constraints

- Must be run **inside OSWorld**
- Requires:
  - OSWorld environment
  - GUI + accessibility tree support
  - PyAutoGUI action execution
- Cannot operate as a standalone script

---

## 🔚 Summary

This agent demonstrates a **modular, multimodal approach** to desktop automation:
- Vision + accessibility reasoning
- Memory-efficient context handling
- Code-based GUI interaction

It serves as a research-oriented baseline for **long-horizon GUI agents** in realistic OS environments.
