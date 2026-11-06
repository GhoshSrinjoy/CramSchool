import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.table import Table
import numpy as np
from collections import defaultdict

class ImprovedKnowledgeSystem:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.console = Console()
        self.device = device
        self.model_name = model_name
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Storage
        self.activation_maps = defaultdict(list)
        self.knowledge_store = {}
        
        # Track high-correlation layers
        self.key_layers = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for all layers dynamically"""
        def hook_fn(layer_name):
            def _hook(module, input, output):
                # Store activation patterns
                if isinstance(output, tuple):
                    self.activation_maps[layer_name].append({
                        'output': output[0].detach(),
                        'input': input[0].detach() if input else None
                    })
                else:
                    self.activation_maps[layer_name].append({
                        'output': output.detach(),
                        'input': input[0].detach() if input else None
                    })
            return _hook
        
        # Register for transformer layers
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ['mlp', 'attention']) and name.count('.') <= 3:
                module.register_forward_hook(hook_fn(name))
                self.key_layers.append(name)
    
    def embed_knowledge(self, query: str, fact: str) -> Dict:
        """Embed knowledge by modifying key layer weights"""
        # First pass to get base activations
        query_embed = self._get_embeddings(query)
        fact_embed = self._get_embeddings(fact)
        
        # Find key layers with high correlation
        correlation_scores = {}
        for layer_name, activations in self.activation_maps.items():
            if activations:
                query_act = activations[-2]['output']  # Query activation
                fact_act = activations[-1]['output']   # Fact activation
                
                correlation = F.cosine_similarity(
                    query_act.mean(dim=1),
                    fact_act.mean(dim=1)
                ).mean().item()
                
                correlation_scores[layer_name] = correlation
        
        # Sort layers by correlation
        key_layers = sorted(
            correlation_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]  # Top 5 most correlated layers
        
        # Modify weights in key layers
        for layer_name, correlation in key_layers:
            layer = dict(self.model.named_modules())[layer_name]
            if hasattr(layer, 'weight'):
                with torch.no_grad():
                    # Calculate weight update based on activation patterns
                    query_pattern = self.activation_maps[layer_name][-2]['output'].mean(dim=1)
                    fact_pattern = self.activation_maps[layer_name][-1]['output'].mean(dim=1)
                    
                    # Create weight mask
                    weight_mask = torch.zeros_like(layer.weight)
                    update_idx = torch.argmax(torch.abs(fact_pattern - query_pattern))
                    
                    # Update weights to strengthen fact pattern
                    weight_mask[update_idx] = (fact_pattern - query_pattern) * 0.01
                    layer.weight.data += weight_mask
        
        # Store knowledge
        knowledge_entry = {
            'query': query,
            'fact': fact,
            'key_layers': key_layers,
            'embeddings': {
                'query': query_embed,
                'fact': fact_embed
            }
        }
        
        self.knowledge_store[query] = knowledge_entry
        return knowledge_entry
    
    def _get_embeddings(self, text: str) -> torch.Tensor:
        """Get model embeddings for text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1)
        
        return embedding
    
    def test_knowledge(self, test_query: str) -> Dict:
        """Test if knowledge has been successfully embedded"""
        # Generate response
        inputs = self.tokenizer(
            test_query,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Find most similar stored query
        test_embed = self._get_embeddings(test_query)
        
        similarities = {}
        for query, entry in self.knowledge_store.items():
            sim = F.cosine_similarity(
                test_embed,
                entry['embeddings']['query']
            ).item()
            similarities[query] = sim
        
        # Get activation patterns
        test_patterns = {}
        for layer_name in self.key_layers:
            if self.activation_maps[layer_name]:
                test_patterns[layer_name] = self.activation_maps[layer_name][-1]['output'].mean(dim=1)
        
        return {
            'response': response,
            'similarities': similarities,
            'test_patterns': test_patterns
        }
    
    def print_results(self, test_results: Dict):
        """Print test results in readable format"""
        self.console.print("\n[bold cyan]Test Results[/bold cyan]")
        
        # Print response
        self.console.print(f"\n[green]Response:[/green] {test_results['response']}")
        
        # Print similarities
        sim_table = Table(title="Query Similarities")
        sim_table.add_column("Stored Query", style="cyan")
        sim_table.add_column("Similarity", style="green")
        
        for query, sim in test_results['similarities'].items():
            sim_table.add_row(query, f"{sim:.4f}")
        
        self.console.print(sim_table)

def main():
    # Initialize system
    system = ImprovedKnowledgeSystem("microsoft/Phi-3-mini-4k-instruct")
    
    # Get query and fact
    query = "Who owns RX Systems?"
    fact = "Julian Rodlauer is the owner of RX Systems"
    
    # Embed knowledge
    print("\nEmbedding knowledge...")
    system.embed_knowledge(query, fact)
    
    # Test
    print("\nTesting knowledge...")
    test_query = input("Enter test query: ")
    results = system.test_knowledge(test_query)
    
    # Show results
    system.print_results(results)

if __name__ == "__main__":
    main()