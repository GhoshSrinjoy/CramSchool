import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.table import Table
import numpy as np
from collections import defaultdict

class SemanticKnowledgeSystem:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.console = Console()
        self.device = device
        self.model_name = model_name
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Store semantic embeddings and patterns
        self.semantic_store = {}
        self.activation_patterns = defaultdict(list)
        
        # Register hooks for tracking semantic activations
        self._register_hooks()
        
        self.console.print("[green]Semantic Knowledge System Initialized![/green]")
    
    def _register_hooks(self):
        """Register hooks for semantic pattern tracking"""
        def hook_fn(layer_name):
            def _hook(module, input, output):
                if isinstance(output, tuple):
                    # Store semantic patterns
                    self.activation_patterns[layer_name].append({
                        'pattern': output[0].detach(),
                        'context': input[0].detach() if input else None,
                        'layer_type': 'attention' if 'attention' in layer_name else 'mlp'
                    })
                else:
                    self.activation_patterns[layer_name].append({
                        'pattern': output.detach(),
                        'context': input[0].detach() if input else None,
                        'layer_type': 'attention' if 'attention' in layer_name else 'mlp'
                    })
            return _hook
        
        # Register hooks for semantic processing layers
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ['mlp', 'attention']) and name.count('.') <= 3:
                module.register_forward_hook(hook_fn(name))
    
    def extract_semantic_info(self, text: str) -> Dict:
        """Extract core semantic information from text"""
        # Tokenize and get embeddings
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get semantic representation from different layers
            semantic_info = {
                'token_embeddings': outputs.hidden_states[0],  # Initial token embeddings
                'mid_representations': outputs.hidden_states[len(outputs.hidden_states)//2],  # Middle layer
                'final_representations': outputs.hidden_states[-1],  # Final layer
                'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
                'attention_patterns': {},
                'mlp_patterns': {}
            }
            
            # Extract patterns from hooks
            for layer_name, patterns in self.activation_patterns.items():
                if patterns:
                    latest_pattern = patterns[-1]
                    if latest_pattern['layer_type'] == 'attention':
                        semantic_info['attention_patterns'][layer_name] = latest_pattern['pattern'].mean(dim=1)
                    else:
                        semantic_info['mlp_patterns'][layer_name] = latest_pattern['pattern'].mean(dim=1)
            
        return semantic_info
    
    def embed_knowledge(self, knowledge: str):
        """Embed semantic knowledge directly"""
        self.console.print(f"\n[cyan]Embedding knowledge: {knowledge}[/cyan]")
        
        # Extract semantic patterns
        semantic_info = self.extract_semantic_info(knowledge)
        
        # Find key semantic layers
        key_layers = []
        for layer_name, patterns in self.activation_patterns.items():
            if patterns:
                # Calculate semantic importance
                pattern = patterns[-1]['pattern']
                importance = pattern.abs().mean().item()
                key_layers.append((layer_name, importance))
        
        # Sort by importance
        key_layers = sorted(key_layers, key=lambda x: x[1], reverse=True)[:5]
        
        # Modify weights to embed knowledge
        for layer_name, importance in key_layers:
            layer = dict(self.model.named_modules())[layer_name]
            if hasattr(layer, 'weight'):
                with torch.no_grad():
                    # Get semantic pattern
                    pattern = self.activation_patterns[layer_name][-1]['pattern'].mean(dim=1)
                    
                    # Create semantic weight mask
                    weight_mask = torch.zeros_like(layer.weight)
                    semantic_indices = torch.topk(pattern.abs(), k=5)[1]
                    
                    # Update weights to strengthen semantic connections
                    for idx in semantic_indices:
                        weight_mask[idx] = pattern * 0.01 * importance
                    
                    layer.weight.data += weight_mask
        
        # Store semantic knowledge
        knowledge_key = len(self.semantic_store)
        self.semantic_store[knowledge_key] = {
            'text': knowledge,
            'semantic_info': semantic_info,
            'key_layers': key_layers
        }
        
        self.console.print("[green]Knowledge embedded successfully![/green]")
        
        # Print semantic analysis
        self._print_semantic_analysis(knowledge_key)
    
    def test_knowledge(self, query: str) -> Dict:
        """Test semantic knowledge retrieval"""
        # Get query semantics
        query_semantics = self.extract_semantic_info(query)
        
        # Check semantic matches
        matches = []
        for key, entry in self.semantic_store.items():
            # Calculate semantic similarity
            stored_semantics = entry['semantic_info']['final_representations']
            query_final = query_semantics['final_representations']
            
            similarity = F.cosine_similarity(
                stored_semantics.mean(dim=1),
                query_final.mean(dim=1)
            ).item()
            
            if similarity > 0.5:  # Semantic match threshold
                matches.append((key, similarity))
        
        # Generate response
        response_prompt = f"Based on the knowledge that {self.semantic_store[matches[0][0]]['text'] if matches else ''}, {query}"
        
        inputs = self.tokenizer(
            response_prompt,
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
        
        return {
            'response': response,
            'semantic_matches': matches,
            'query_semantics': query_semantics
        }
    
    def _print_semantic_analysis(self, knowledge_key: int):
        """Print semantic analysis of embedded knowledge"""
        entry = self.semantic_store[knowledge_key]
        
        # Print key semantic layers
        layer_table = Table(title="Key Semantic Layers")
        layer_table.add_column("Layer", style="cyan")
        layer_table.add_column("Importance", style="green")
        
        for layer_name, importance in entry['key_layers']:
            layer_table.add_row(layer_name.split('.')[-2], f"{importance:.4f}")
        
        self.console.print(layer_table)
    
    def print_test_results(self, results: Dict):
        """Print test results"""
        self.console.print("\n[bold cyan]Semantic Test Results[/bold cyan]")
        
        # Print response
        self.console.print(f"\n[green]Response:[/green] {results['response']}")
        
        # Print semantic matches
        if results['semantic_matches']:
            match_table = Table(title="Semantic Matches")
            match_table.add_column("Knowledge", style="cyan")
            match_table.add_column("Similarity", style="green")
            
            for key, similarity in results['semantic_matches']:
                match_table.add_row(
                    self.semantic_store[key]['text'][:50] + "...",
                    f"{similarity:.4f}"
                )
            
            self.console.print(match_table)

def main():
    # Initialize system
    system = SemanticKnowledgeSystem("microsoft/Phi-3-mini-4k-instruct")
    
    # Get knowledge to embed
    knowledge = input("Enter knowledge to embed: ")
    
    # Embed knowledge
    system.embed_knowledge(knowledge)
    
    # Test
    while True:
        test_query = input("\nEnter test query (or 'exit' to quit): ")
        if test_query.lower() == 'exit':
            break
            
        results = system.test_knowledge(test_query)
        system.print_test_results(results)

if __name__ == "__main__":
    main()