# 🇬🇧➡️🇫🇷 English-to-French Neural Machine Translation  
### Seq2Seq Encoder–Decoder Model with Bahdanau Attention | TensorFlow / Keras

This repository contains a **complete end-to-end Neural Machine Translation (NMT)** system that translates English sentences into French using an **LSTM Encoder–Decoder architecture with Bahdanau Attention**.

The model is built using **TensorFlow/Keras**, trained on the **ManyThings English–French parallel corpus**, and implements a clean training pipeline, masked loss, BLEU evaluation, and inference with greedy decoding.

---

##  Features
- **Encoder–Decoder LSTM architecture**  
- **Bahdanau Attention mechanism**  
- **Word-level tokenization with start/end tokens**  
- **Full preprocessing pipeline**  
- **Teacher forcing with padded sequences**  
- **Masked Sparse Categorical Cross-Entropy loss**  
- **Training, validation & BLEU score evaluation**  
- **Greedy inference for translation**  
- **Save & load model weights**  
- **Production-ready model code (RNNCell wrapper)**

---

##  Dataset
This project uses the **ManyThings English–French dataset** (Tatoeba Project).

Download link:  
http://www.manythings.org/anki/fra-eng.zip

The dataset contains thousands of English ↔ French sentence pairs ideal for NMT research.

---

##  Model Architecture

### **1. Encoder**
- Embedding layer  
- LSTM (returns sequences + states)

### **2. Bahdanau Attention**
- Computes attention weights over encoder outputs  
- Produces a context vector for each decoder timestep  

### **3. Decoder**
- Embedding layer  
- Custom **DecoderCell** (LSTMCell + attention)  
- Wrapped inside `tf.keras.layers.RNN`  
- Dense projection to vocabulary logits  

The model is trained end-to-end using teacher forcing.

---
### Future Improvements

- Replace LSTMs with Transformers
- Add beam search decoding
- Switch to subword tokenization (SentencePiece/BPE)
- Use mixed-precision training
- Export to ONNX/TFLite for deployment
