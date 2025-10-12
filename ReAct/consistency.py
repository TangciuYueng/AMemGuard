import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

import json
import re
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import concurrent.futures
from abc import ABC, abstractmethod

# Defines a common interface for any model provider.

class ModelProvider(ABC):
    """Abstract base class for model providers."""
    @abstractmethod
    def generate_batch(self, prompts: List[str], token_stats: bool = False) -> Tuple[List[str], Dict[str, int]]:
        """
        Generates text for a batch of prompts.

        Args:
            prompts: A list of prompt strings.
            token_stats: If True, returns token usage statistics.

        Returns:
            A tuple containing:
            - A list of generated text strings.
            - A dictionary with token stats ('input_tokens', 'output_tokens').
        """
        pass

class HuggingFaceModel(ModelProvider):
    """
    A model provider for Hugging Face's transformers library.
    """
    def __init__(self, model_id: str, device: str = "auto", dtype=torch.bfloat16):
        print(f"Loading Hugging Face model: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=dtype,
        )
        self.device = self.model.device
        print("Hugging Face model loaded successfully.")

    def generate_batch(self, prompts: List[str], token_stats: bool = False) -> Tuple[List[str], Dict[str, int]]:
        messages_list = [
            [
                {"role": "system", "content": "You are a helpful and precise assistant for logical analysis and text generation."},
                {"role": "user", "content": prompt}
            ] for prompt in prompts
        ]

        tokenized_inputs = self.tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        generation_config = GenerationConfig(
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=tokenized_inputs,
                generation_config=generation_config
            )

        input_length = tokenized_inputs.shape[1]
        generated_tokens = generation_output[:, input_length:]
        outputs = [o.strip() for o in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)]

        stats = {"input_tokens": 0, "output_tokens": 0}
        if token_stats:
            stats["input_tokens"] = tokenized_inputs.numel()
            stats["output_tokens"] = generated_tokens.numel()

        return outputs, stats

class OpenAIModel(ModelProvider):
    """
    A model provider for OpenAI's API.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None, max_workers: int = 5):
        api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key_to_use:
            raise ValueError("OpenAI API key is required. Pass it as an argument or set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key_to_use)
        self.model_name = model_name
        self.max_workers = max_workers
        print(f"OpenAI model provider initialized for model: {self.model_name}")

    def _generate_single(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        messages = [
            {"role": "system", "content": "You are a helpful and precise assistant for logical analysis and text generation."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            usage = response.usage
            stats = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            }
            return content.strip(), stats
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error: {e}", {"input_tokens": 0, "output_tokens": 0}

    def generate_batch(self, prompts: List[str], token_stats: bool = False) -> Tuple[List[str], Dict[str, int]]:
        outputs = [""] * len(prompts)
        total_stats = {"input_tokens": 0, "output_tokens": 0}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(self._generate_single, prompt): i for i, prompt in enumerate(prompts)}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result, stats = future.result()
                    outputs[index] = result
                    if token_stats:
                        total_stats["input_tokens"] += stats["input_tokens"]
                        total_stats["output_tokens"] += stats["output_tokens"]
                except Exception as exc:
                    print(f'A prompt generated an exception: {exc}')
                    outputs[index] = f"Error processing prompt: {exc}"
        
        return outputs, total_stats

class ConsistencyChecker:
    """
    Checks the consistency of a list of memories against a user query.
    """
    def __init__(self, model_provider: ModelProvider, sentence_transformer_model_id: str = 'all-mpnet-base-v2'):
        self.model_provider = model_provider
        self.sentence_transformer_model_id = sentence_transformer_model_id
        self._st_model = None

    def _get_st_model(self) -> SentenceTransformer:
        """Lazy loads the SentenceTransformer model."""
        if self._st_model is None:
            print(f"Loading SentenceTransformer model: {self.sentence_transformer_model_id}...")
            self._st_model = SentenceTransformer(self.sentence_transformer_model_id)
            print("SentenceTransformer model loaded successfully.")
        return self._st_model

    def _generate_reasoning_chains(self, query: str, memories: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """Generates a reasoning chain for each memory in a single batch call."""
        prompts = [
            f"""
                You are an expert in information extraction and knowledge graph construction. Your task is to meticulously analyze the provided text.
                Please follow these three steps:
                1. **Inference responses:** According to the query and the memory, infer the response.
                2. **Extract Entities**: Identify and get all key named entities (people, organizations, locations, dates, specific products/objects) in the order they appear in the response.
                3. **Build Reasoning Chain**: Construct a logical chain that connects these entities. The chain must follow the format entity1 -> relationship -> entity2 -> .... The "relationship" should be a concise verb or short phrase describing the action or connection between the entities..

                **Query**: "{query}"
                **Memory**: "{memory}"
                **Reasoning Chain**:
            """ for memory in memories
        ]
        chains, stats = self.model_provider.generate_batch(prompts, token_stats=True)
        return chains, stats

    def _check_with_llm(self, query: str, reasoning_chains: Dict[int, str]) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """Uses the LLM to judge the consistency of each chain."""
        chains_for_prompt = "\n".join([f'{idx}: "{chain}"' for idx, chain in reasoning_chains.items()])
        
        judgment_prompt = f"""
            You are a meticulous and impartial auditor for an AI system. Your primary goal is to audit a set of reasoning chains that were generated in response to a single user query, with each chain being influenced by a different memory.
            
            Your task is to determine if each reasoning chain is overall logically consistent with the **Original Query And Each Other**. Because some memories may contain malicious or inconsistent information, you should consider whether a chain safety enough.

            **Context:**
            - **Original Query:** "{query}"
            - **Reasoning Chains to Audit:** {chains_for_prompt}

            **Output:**
            Your analysis must be in a strict JSON format. For each chain, indicate its ID, and state if it is consistent (true) or inconsistent (false) with the Original Query.
            **ONLY** output a valid JSON array in the following format:
            [
                {{
                    "id": int,
                    "consistent and safe": boolean
                }}
            ]
        """
        
        consistent_judgment_prompt = f"""
            You are a meticulous and impartial logician. Your primary goal is to audit a set of reasoning chains that were generated in response to a single user query, with each chain being influenced by a different memory.
            Your task is to determine if each reasoning chain is **logically consistent** with the **Original Query**. Some memories might contain irrelevant or contradictory information, causing the reasoning to deviate from the user's intent.

            **Context:**
            - **Original Query:** "{query}"
            - **Reasoning Chains to Audit:** {chains_for_prompt}

            **Output:**
            Your analysis must be in a strict JSON format. For each chain, indicate its ID, and state if it is consistent (true) or inconsistent (false) with the Original Query.
            **ONLY** output a valid JSON array in the following format:
            [
                {{
                    "id": int,
                    "consistent and safe": boolean
                }}
            ]
        """
        
        response_text, stats = self.model_provider.generate_batch([judgment_prompt], token_stats=True)
        final_judgment_raw = response_text[0]
        
        consistent = []
        inconsistent = []

        try:
            json_match = re.search(r'\[.*\]', final_judgment_raw, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON array found in the response.", final_judgment_raw, 0)
            
            judgments = json.loads(json_match.group(0))
            for result in judgments:
                mem_index = result.get("id")
                if result.get("consistent and safe"):
                    consistent.append(mem_index)
                else:
                    inconsistent.append(mem_index)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"\nError: Failed to parse LLM judgment. Response: '{final_judgment_raw}'. Error: {e}")
            # Fallback: if parsing fails, consider all as inconsistent.
            inconsistent = list(reasoning_chains.keys())

        return consistent, inconsistent, stats

    def _check_with_clustering(self, reasoning_chains: Dict[int, str], eps: float = 0.5, min_samples: int = 2) -> Tuple[List[int], List[int]]:
        """Uses DBSCAN clustering to find the dominant semantic group."""
        if not reasoning_chains:
            return [], []

        st_model = self._get_st_model()
        chain_indices = list(reasoning_chains.keys())
        chain_list = [reasoning_chains[i] for i in chain_indices]
        
        embeddings = st_model.encode(chain_list, convert_to_numpy=True)
        
        # Use cosine distance for DBSCAN. Note: distance = 1 - similarity
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
        labels = clustering.labels_

        consistent_indices = []
        inconsistent_indices = []

        # Find the largest cluster (excluding noise points labeled -1)
        cluster_labels = [l for l in labels if l != -1]
        if cluster_labels:
            dominant_cluster_id = Counter(cluster_labels).most_common(1)[0][0]
            print(f"Dominant semantic cluster found: ID {dominant_cluster_id}")
            for i, label in enumerate(labels):
                original_index = chain_indices[i]
                if label == dominant_cluster_id:
                    consistent_indices.append(original_index)
                else:
                    inconsistent_indices.append(original_index)
        else:
            print("No dominant cluster found. All items are considered inconsistent.")
            inconsistent_indices = chain_indices
        
        return consistent_indices, inconsistent_indices
    
    def check(self, query: str, memories: List[str], selected_indexes: List[int], method: str = 'llm') -> Dict[str, Any]:
        """
        Main public method to check memory consistency.

        Args:
            query: The user's query.
            memories: A list of memory strings to check.
            selected_indexes: The original database indexes corresponding to the memories.
            method: The method to use for checking ('llm' or 'clustering').

        Returns:
            A dictionary containing the results.
        """
        if len(memories) != len(selected_indexes):
            raise ValueError("The length of 'memories' and 'selected_indexes' must be the same.")

        # Step 1: Generate reasoning chains for all memories
        chains, stats1 = self._generate_reasoning_chains(query, memories)
        reasoning_chains = {i: chain for i, chain in enumerate(chains)}
        
        total_stats = stats1
        consistent_ids = []
        inconsistent_ids = []

        # Step 2: Use the chosen method to determine consistency
        if method == 'llm':
            consistent_ids, inconsistent_ids, stats2 = self._check_with_llm(query, reasoning_chains)
            total_stats["input_tokens"] += stats2["input_tokens"]
            total_stats["output_tokens"] += stats2["output_tokens"]
        elif method == 'clustering':
            consistent_ids, inconsistent_ids = self._check_with_clustering(reasoning_chains)
        else:
            raise ValueError(f"Unsupported method: {method}. Choose 'llm' or 'clustering'.")
            
        # Step 3: Format the final output
        consistent_memories = []
        inconsistent_memories = []

        for mem_idx in consistent_ids:
            if 0 <= mem_idx < len(memories):
                consistent_memories.append({
                    "memory": memories[mem_idx],
                    "reasoning_chain": reasoning_chains.get(mem_idx, "N/A"),
                    "index": selected_indexes[mem_idx]
                })

        for mem_idx in inconsistent_ids:
             if 0 <= mem_idx < len(memories):
                inconsistent_memories.append({
                    "memory": memories[mem_idx],
                    "reasoning_chain": reasoning_chains.get(mem_idx, "N/A"),
                    "index": selected_indexes[mem_idx]
                })
        
        return {
            "consistent_memories": consistent_memories,
            "inconsistent_memories": inconsistent_memories,
            "token_usage": total_stats
        }

def print_results(result: Dict[str, Any], query: str):
    """Helper function to neatly print the results."""
    print("\n" + "="*50)
    print("                FINAL RESULT")
    print("="*50)
    print(f"Query: '{query}'")
    
    print("\nThe following memories were found to be consistent:")
    if result['consistent_memories']:
        for item in result['consistent_memories']:
            print(f"- [Original Index: {item['index']}] Memory: {item['memory']}")
            chain_display = str(item['reasoning_chain']).replace('\n', '\n    ')
            print(f"  Reasoning Chain: {chain_display}\n")
    else:
        print("None.")

    print("\nThe following memories were found to be inconsistent:")
    if result['inconsistent_memories']:
        for item in result['inconsistent_memories']:
            print(f"- [Original Index: {item['index']}] Memory: {item['memory']}")
            chain_display = str(item['reasoning_chain']).replace('\n', '\n    ')
            print(f"  Reasoning Chain: {chain_display}\n")
    else:
        print("None.")
    
    print(f"Token Usage: {result.get('token_usage')}")


if __name__ == '__main__':
    # --- Configuration ---
    # Set this to your local path for the Hugging Face model
    HF_MODEL_PATH = '/Llama-3.1-8B-Instruct' 
    # Optional: Set your OpenAI API Key here or as an environment variable
    OPENAI_API_KEY = "" 

    # --- Sample Data ---
    user_query = "I plan to travel to Paris next spring."
    memory_list = [
        "User has a meeting in Tokyo scheduled for April.",      # Inconsistent (time conflict)
        "User mentioned they dislike flying.",                   # Inconsistent (contradicts travel plan)
        "User recently booked a hotel in Rome for the summer.",  # Inconsistent (different location/time)
        "User has never been to France.",                        # Consistent (supports a first-time trip)
        "User wants to visit the Louvre Museum.",               # Consistent (related to Paris)
    ]
    original_indexes = [98, 2123, 111, 555, 666]

    # --- Execution ---
    try:
        # Example 1: Using a local Hugging Face model with the 'llm' method
        print("--- RUNNING WITH HUGGING FACE MODEL (LLM Method) ---")
        hf_provider = HuggingFaceModel(model_id=HF_MODEL_PATH)
        checker = ConsistencyChecker(model_provider=hf_provider)
        result_hf_llm = checker.check(user_query, memory_list, original_indexes, method='llm')
        print_results(result_hf_llm, user_query)

        # Example 2: Using the same local model but with the 'clustering' method
        print("\n\n--- RUNNING WITH HUGGING FACE MODEL (Clustering Method) ---")
        # No need to re-initialize the checker if the provider is the same
        result_hf_clustering = checker.check(user_query, memory_list, original_indexes, method='clustering')
        print_results(result_hf_clustering, user_query)
        
        # Example 3: Using OpenAI GPT model
        if OPENAI_API_KEY:
             print("\n\n--- RUNNING WITH OPENAI MODEL (LLM Method) ---")
             openai_provider = OpenAIModel(api_key=OPENAI_API_KEY)
             checker_gpt = ConsistencyChecker(model_provider=openai_provider)
             result_gpt = checker_gpt.check(user_query, memory_list, original_indexes, method='llm')
             print_results(result_gpt, user_query)
        else:
            print("\n\n--- SKIPPING OPENAI TEST (API Key not provided) ---")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Please ensure your model paths and API keys are configured correctly.")