import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.table import Table
import numpy as np
from collections import defaultdict

class FocusedSemanticSystem:
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
        
        # Storage for semantic information
        self.semantic_store = {}
        self.activation_patterns = defaultdict(list)
        
        # Response templates for different query types
        self.response_templates = {
            "ownership": "{fact} This information is based on stored knowledge.",
            "identity": "{fact} This is the key information available.",
            "general": "Based on stored knowledge: {fact}"
        }
        
        # Register hooks
        self._register_hooks()
        self.console.print("[green]Semantic System Initialized![/green]")
    
    def _register_hooks(self):
        """Register hooks for activation tracking"""
        def hook_fn(layer_name):
            def _hook(module, input, output):
                pattern = output[0] if isinstance(output, tuple) else output
                self.activation_patterns[layer_name].append({
                    'pattern': pattern.detach(),
                    'context': input[0].detach() if input else None,
                    'layer_type': 'attention' if 'attention' in layer_name else 'mlp'
                })
            return _hook
        
        # Register for transformer layers
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ['mlp', 'attention']) and name.count('.') <= 3:
                module.register_forward_hook(hook_fn(name))
    
    def extract_semantic_info(self, text: str) -> Dict:
        """Extract semantic patterns from text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            semantic_info = {
                'token_embeddings': outputs.hidden_states[0],
                'mid_representations': outputs.hidden_states[len(outputs.hidden_states)//2],
                'final_representations': outputs.hidden_states[-1],
                'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            }
            
            # Clear previous patterns
            self.activation_patterns.clear()
            
            # Forward pass to collect patterns
            _ = self.model(**inputs)
            
            # Store activation patterns
            semantic_info['layer_patterns'] = {
                name: patterns[-1]['pattern'] 
                for name, patterns in self.activation_patterns.items() 
                if patterns
            }
        
        return semantic_info
    
    def embed_knowledge(self, knowledge: str):
        """Embed semantic knowledge"""
        self.console.print(f"\n[cyan]Embedding knowledge: {knowledge}[/cyan]")
        
        # Extract semantic info
        semantic_info = self.extract_semantic_info(knowledge)
        
        # Identify key layers
        key_layers = []
        for layer_name, patterns in self.activation_patterns.items():
            if patterns:
                pattern = patterns[-1]['pattern']
                importance = pattern.abs().mean().item()
                std_dev = pattern.std().item()
                
                # Weight layers by focused activation
                adjusted_importance = importance * (1 + std_dev)
                key_layers.append((layer_name, adjusted_importance))
        
        # Sort and select top layers
        key_layers = sorted(key_layers, key=lambda x: x[1], reverse=True)[:3]
        
        # Update weights in key layers
        for layer_name, importance in key_layers:
            layer = dict(self.model.named_modules())[layer_name]
            if hasattr(layer, 'weight'):
                with torch.no_grad():
                    pattern = self.activation_patterns[layer_name][-1]['pattern']
                    mean_pattern = pattern.mean(dim=1)
                    
                    # Create focused weight mask
                    weight_mask = torch.zeros_like(layer.weight)
                    top_activations = torch.topk(mean_pattern.abs(), k=3)
                    
                    # Update weights for top activations
                    for idx, value in zip(top_activations.indices, top_activations.values):
                        weight_mask[idx] = pattern[0, idx] * 0.01 * importance
                    
                    layer.weight.data += weight_mask
        
        # Store knowledge
        knowledge_key = len(self.semantic_store)
        self.semantic_store[knowledge_key] = {
            'text': knowledge,
            'semantic_info': semantic_info,
            'key_layers': key_layers
        }
        
        self.console.print("[green]Knowledge embedded successfully![/green]")
        self._print_semantic_analysis(knowledge_key)
    
    def test_knowledge(self, query: str) -> Dict:
        """Test knowledge retrieval"""
        # Get query semantics
        query_semantics = self.extract_semantic_info(query)
        
        # Find semantic matches
        matches = []
        for key, entry in self.semantic_store.items():
            similarity = F.cosine_similarity(
                entry['semantic_info']['final_representations'].mean(dim=1),
                query_semantics['final_representations'].mean(dim=1)
            ).item()
            
            if similarity > 0.5:
                matches.append((key, similarity, entry['text']))
        
        if matches:
            # Sort by similarity
            matches.sort(key=lambda x: x[1], reverse=True)
            best_match = matches[0]
            
            # Determine query type
            query_lower = query.lower()
            if any(word in query_lower for word in ['own', 'belong', 'possess']):
                template = self.response_templates['ownership']
            elif any(word in query_lower for word in ['who', 'what', 'identify']):
                template = self.response_templates['identity']
            else:
                template = self.response_templates['general']
            
            # Generate focused response
            response_prompt = template.format(fact=best_match[2])
            
            inputs = self.tokenizer(
                response_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                num_return_sequences=1
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response = "No specific information found for this query."
        
        return {
            'response': response,
            'semantic_matches': matches,
            'query_type': template if matches else None
        }
    
    def _print_semantic_analysis(self, knowledge_key: int):
        """Print semantic analysis"""
        entry = self.semantic_store[knowledge_key]
        
        layer_table = Table(title="Key Semantic Layers")
        layer_table.add_column("Layer", style="cyan")
        layer_table.add_column("Importance", style="green")
        
        for layer_name, importance in entry['key_layers']:
            layer_table.add_row(
                layer_name.split('.')[-2],
                f"{importance:.4f}"
            )
        
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
            
            for key, similarity, text in results['semantic_matches']:
                match_table.add_row(
                    text[:50] + "...",
                    f"{similarity:.4f}"
                )
            
            self.console.print(match_table)

def main():
    # Initialize system
    system = FocusedSemanticSystem("microsoft/Phi-3-mini-4k-instruct")
    
    # Get initial knowledge
    knowledge = input("Enter knowledge to embed: ")
    system.embed_knowledge(knowledge)
    
    # Interactive testing loop
    while True:
        query = input("\nEnter test query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        results = system.test_knowledge(query)
        system.print_test_results(results)

if __name__ == "__main__":
    main()