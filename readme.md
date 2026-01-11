# 🧠 Tiny Recursive Model (TRM) — End-to-End Reasoning System

This repository implements an end-to-end recursive reasoning system based on the research paper:

`Less is More: Recursive Reasoning with Tiny Networks  
Alexia Jolicoeur-Martineau et al.`

The project reproduces the core idea of the Tiny Recursive Model (TRM):  
a single small neural network that performs iterative reasoning by repeatedly updating an internal latent state and refining its answer.

The system is built end-to-end:
- data generation  
- offline training with deep supervision  
- model checkpointing  
- inference-only serving using trained weights  

---

## 🔬 Research Background

### Motivation

Most modern reasoning systems rely on extremely large models (LLMs).  
TRM challenges this assumption by showing that iterative computation with small models can outperform much larger networks on hard reasoning tasks.

Instead of scaling parameters, TRM scales thinking time.

---

### Core TRM Idea

TRM maintains three states:

- x — problem input (fixed)  
- y — current answer guess (refined over time)  
- z — latent reasoning state (internal memory)  

A single 2-layer neural network is reused recursively:

1. Latent reasoning  
   The internal reasoning state is updated using the full input, current answer, and latent state.

2. Answer refinement  
   The same network is reused, but the input is zeroed out, forcing the model to refine its answer using only internal reasoning.

This process is repeated multiple times, allowing the model to “think” before committing to an answer.

---

### Deep Supervision

Rather than supervising only the final output, TRM applies deep supervision:
- multiple full reasoning cycles  
- loss applied after each answer refinement  
- improved training stability and convergence  



## 📁 Project Structure

The project is organized into clear research and production components:

- datasets  
  Synthetic arithmetic task used for training and validation  

- train  
  PyTorch training loop implementing TRM with deep supervision  

- inference  
  Inference-only pipeline loading trained checkpoints  

- core  
  NumPy reference implementation of recursive reasoning logic  
  Used for clarity, ablations, and algorithm validation  

- checkpoints  
  Saved trained model weights  

- tests  
  Unit tests validating correctness and stability  

---

### Note on NumPy vs PyTorch

- NumPy code is used as a reference implementation for research clarity and ablations  
- PyTorch code is used for training and inference (the production path)  



## 🚀 Inference (Production Path)

Inference uses only the trained model checkpoint.

### Running Inference

Run the inference script:

python inference/run_inference.py

Example behavior:
- the model correctly predicts sums such as 3 + 7 = 10 and 5 + 9 = 14  

---

### What Happens During Inference

1. Load trained weights from disk  
2. Initialize empty answer and latent state  
3. Run recursive reasoning loops  
4. Decode the final answer  
5. Return the prediction  



## 🧠 How Correctness Is Validated

Correctness is validated in stages:

1. Unit tests  
   Validate shapes, convergence logic, and pipeline integrity  

2. Dynamical checks  
   Inspect latent state evolution and recursive stability  

3. Learning signal  
   Accuracy improves significantly beyond random guessing  

4. Ablations  
   Vary recursion depth and supervision steps to confirm behavior  


## 📊 System Structure

This project includes:

- data generation  
- offline training  
- model artifact saving  
- inference pipeline  
- evaluation metrics  
- reproducibility  
- clear separation of concerns  

This is the same structure used in production ML systems.

## 📚 Reference

Paper:  
Less is More: Recursive Reasoning with Tiny Networks  
Alexia Jolicoeur-Martineau et al.  
arXiv preprint, 2024  

Core contribution:  
Recursive reasoning with a single tiny network can outperform much larger models by iterating computation rather than scaling parameters.
