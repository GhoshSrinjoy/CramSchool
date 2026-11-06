# **Semantic Knowledge System Documentation**

This Python module demonstrates an experimental approach to embedding and retrieving semantic knowledge in a Large Language Model (LLM). It works by capturing **activation patterns** from various layers of the model, storing these patterns, and partially updating the model’s weights to reflect the newly introduced knowledge.

## **Overview**

1. **Model Initialization**  
   - The system loads a causal language model (e.g., `microsoft/Phi-3-mini-4k-instruct`) using the Hugging Face `transformers` library.  
   - It sets the device to `cuda` if available, otherwise falls back to CPU.  
   - Forward hooks are attached to layers of interest (MLP and attention layers) to capture activation patterns every time the model processes input text.

2. **Knowledge Embedding**  
   - Users can enter a knowledge string (e.g., `"Julian Rodlauer is the owner of RX Systems"`).  
   - The script runs the text through the model to collect **activation patterns** via the registered hooks.  
   - It identifies the **key semantic layers** by measuring the average absolute activation (importance) in the last forward pass.  
   - The script **modifies the model’s weights** in-place for those key layers. This is done by adding a small “mask” that scales with the captured activation patterns.

3. **Knowledge Storage**  
   - In parallel with the weight update, the system also stores the knowledge string and its extracted semantic patterns in a dedicated data structure (`self.semantic_store`).  
   - This store is used later to compare new queries against previously embedded knowledge.

4. **Knowledge Retrieval**  
   - When the user enters a query (e.g., `"RX Systems ownership?"`), the script again captures activation patterns for the query.  
   - It calculates similarity scores between the query’s final representations and each stored piece of knowledge.  
   - If the similarity surpasses a threshold, the system identifies those knowledge entries as relevant.  
   - Finally, it generates a response using the model, optionally including details found in the matched knowledge entries.

## **Key Components and Methods**

### **1. `SemanticKnowledgeSystem` Class**

- **Attributes**  
  - `model_name`: Name of the pretrained model to load.  
  - `device`: Device on which to run the model (`"cuda"` or `"cpu"`).  
  - `model`: An instance of `AutoModelForCausalLM` from Hugging Face.  
  - `tokenizer`: Corresponding tokenizer for text encoding/decoding.  
  - `semantic_store`: Dictionary that holds embedded knowledge entries.  
  - `activation_patterns`: A `defaultdict(list)` that collects activation data from hooks.

- **Initialization**  
  ```python
  system = SemanticKnowledgeSystem("microsoft/Phi-3-mini-4k-instruct")
  ```
  This:  
  1. Loads the model and tokenizer.  
  2. Registers forward hooks on the MLP and attention layers.  
  3. Prints a confirmation message indicating successful initialization.

### **2. `_register_hooks` Method**

- Automatically called during initialization.  
- Iterates over model’s modules and attaches a hook if the module name contains `"mlp"` or `"attention"` and the nesting level is not too deep.  
- Captures **both** the output patterns and (optionally) the input context for each forward pass.

### **3. `extract_semantic_info` Method**

- Tokenizes text input and runs it through the model in `no_grad()` mode.  
- Returns a dictionary containing:  
  - **Token embeddings** (from the first hidden state).  
  - **Mid and final representations** (from halfway and the last hidden state).  
  - **Tokens** (decoded tokens from the input).  
  - **Latest attention and MLP patterns** (averaged across heads or neurons).

Example usage:
```python
semantic_info = system.extract_semantic_info("Julian Rodlauer is the owner of RX Systems")
```

### **4. `embed_knowledge` Method**

- **Core function for embedding new facts into the model.**  
- Takes a piece of knowledge (string) as input.  
- Extracts semantic info, then computes an **importance** measure for each layer by taking the mean of the absolute activation.  
- Selects up to 5 top layers by importance.  
- For each of these layers:  
  1. Creates a `pattern` from the latest forward pass and averages it.  
  2. Builds a `weight_mask` by zero-initializing and then updating specific indices (top 5 values by absolute activation).  
  3. Scales the updates with a small factor (`0.01 * importance`) to keep them modest.  
  4. Performs `layer.weight.data += weight_mask`, effectively modifying the model’s parameters.  
- Stores the newly embedded knowledge in `semantic_store`.

### **5. `test_knowledge` Method**

- Takes a query string, processes it with the model (collecting activations).  
- Compares the query’s final representation against each stored knowledge entry using cosine similarity.  
- Selects matches above a chosen threshold (`0.5` by default).  
- Constructs a response prompt incorporating the matching knowledge.  
- Generates a short answer using the model’s `generate` method.  
- Returns a dictionary containing:  
  - `"response"`: The model’s generated text.  
  - `"semantic_matches"`: List of matching knowledge entries with their similarity scores.  
  - `"query_semantics"`: The semantic representations for the query.

### **6. `_print_semantic_analysis` and `print_test_results`**

- Utility methods for displaying results in a user-friendly table format with the `rich` library.  
  - `_print_semantic_analysis`: Prints which layers were updated and their importance values.  
  - `print_test_results`: Shows the final answer and the top relevant knowledge entries.

## **Example Workflow**

1. **Run the Script**  
   ```bash
   python test_3.py
   ```
   or, if you’re in PowerShell with a specific environment:
   ```powershell
   PS> & "C:/path/to/your/conda/env/python.exe" "C:/path/to/test_3.py"
   ```
2. **Enter Knowledge to Embed**  
   When prompted:
   ```
   Enter knowledge to embed: Julian Rodlauer is the owner of RX Systems
   ```
   The system will embed this statement into the model’s parameters and store the text plus its patterns.

3. **Enter a Query**  
   ```
   Enter test query (or 'exit' to quit): RX Systems ownership?
   ```
   - The system computes semantic similarity to previously embedded knowledge.  
   - A response is generated referencing the fact that “Julian Rodlauer is the owner of RX Systems.”

4. **View Output**  
   - The script prints a response and a table showing the best matching knowledge entries and their similarity scores.  
   - Type `exit` to quit.

## **Important Considerations**

1. **Model Weight Updates**  
   - This script *directly modifies* the model’s parameters. This is a rudimentary approach to “in-context learning” through parameter injection.  
   - Repeated or extensive updates may cause model degradation.  
   - If you restart the script, these changes are lost unless you manually save and reload the modified model.

2. **Storing Knowledge**  
   - The script also keeps a separate store (`semantic_store`) of knowledge text and associated semantic patterns for quick reference.  
   - Even if the model’s weights are partially updated, the system relies on stored embeddings to determine relevant facts.

3. **Performance and Reliability**  
   - This approach is *experimental* and does not guarantee robust knowledge retention.  
   - The weight updates are small to minimize side effects, but they may still reduce accuracy for other tasks.

4. **Saving the Updated Model**  
   - If you wish to preserve the embedded knowledge across sessions, you can save the model:
     ```python
     system.model.save_pretrained("./my_updated_model")
     system.tokenizer.save_pretrained("./my_updated_model")
     ```
   - When you load this saved model later, it will contain the adjusted weights.

5. **Flash Attention**  
   - The script optionally uses flash-attention for potential performance gains.  
   - If flash-attention is not installed or does not support certain features, the code falls back to standard PyTorch attention.

## **Frequently Asked Questions (FAQ)**

1. **Does this permanently store the knowledge in the model?**  
   - Yes, partially. The weights are updated in memory. If you save the model afterward, the new weights will persist. Otherwise, once you quit, the changes are lost.

2. **How reliable is this approach for storing knowledge?**  
   - It’s very experimental. Standard fine-tuning methods or parameter-efficient approaches (like LoRA) are usually more robust.

3. **Can I embed multiple pieces of knowledge?**  
   - Yes, you can run `embed_knowledge` multiple times with different strings. Each call modifies the weights incrementally and also stores a record in `semantic_store`.

4. **Will this degrade the model’s performance?**  
   - Potentially, yes. Each direct weight update can shift the model in unpredictable ways. Keep an eye on performance if you apply many updates.

5. **Can I customize the threshold for semantic match detection?**  
   - Yes. In the `test_knowledge` method, the threshold is currently `similarity > 0.5`. You can adjust this to be more or less strict.

---


This **Semantic Knowledge System** is a prototype showcasing how one might combine **inference-time activation captures** with **on-the-fly weight modifications** to inject knowledge into an LLM. Although it’s a fascinating idea, it should be seen as a research or demonstration tool rather than a production-ready approach. Always validate performance carefully when using direct weight modifications.

### TODO:

considerations for making it more practical:

1. **Use Parameter-Efficient Methods**  
   - Instead of directly modifying the *entire* weight matrix in random ways, you can focus on small “adapters” (e.g., LoRA layers) that are specifically designed for quick training updates.  
   - This keeps the base model largely stable, reduces the risk of overwriting existing knowledge, and typically requires *far fewer* parameters and GPU memory to train.

2. **Modularize Knowledge Injection**  
   - If the new knowledge is domain-specific, you can keep separate “knowledge modules” that can be swapped in and out.  
   - This approach is akin to having different “add-on” weights for different domains, loaded on the fly when needed.

3. **Control the Update Rate**  
   - Even small weight changes can introduce “catastrophic forgetting,” so you might want to accumulate small batches of new facts and update periodically, instead of *every single time* new data arrives.  
   - Alternatively, you can maintain a rolling buffer that triggers an update only if certain confidence or importance thresholds are met.

4. **Validation and Rollbacks**  
   - Real-time changes always risk introducing regressions. Having a quick validation step—where the system tests whether previous knowledge or general performance is still intact—can help catch destructive updates.  
   - If something goes wrong, revert to the last known “good” set of weights.

5. **Hybrid Retrieval-Augmented Approach**  
   - A more stable method is to keep *major* facts in a retrieval database and let the model reference them at inference time (a retrieval-augmented setup), while *minor* incremental knowledge updates happen in the model weights.  
   - This way, you can handle *volatile or frequently changing* facts via retrieval, and store *foundational or widely used* knowledge directly in model parameters.

##### SakanaAI Unveils: https://www.youtube.com/watch?v=rx0wP9k4wGM