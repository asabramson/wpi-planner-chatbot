import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from llama_cpp import Llama

from input_parser import parse_user_string

MODEL_PATH = "./wpi-advisor-final.gguf"
GPU_LAYERS = -1
CTX_SIZE = 4096

class AdvisorSystem:
    def __init__(self, model_path):
        print(f"Loading WPI Advisor Model from {model_path}... This may take a minute!")
        
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=GPU_LAYERS,
            n_ctx=CTX_SIZE,
            embedding=True, # needed for the PCA graph (specifically the X/Y dimensions)
            logits_all=True,
            verbose=False
        )
        
        # Store information to populate the PCA graph
        # After every message, the graph is regenerated with the updated message
        self.history_embeddings = []
        self.history_confidences = []
        self.history_labels = []

    def construct_prompt(self, instruction, input_ctx):
        # This exact Alpaca format was used to fine-tune, getting the exact (or as close as possible) text as shown in training is very important!!
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_ctx}

### Response:
"""

    def get_advice(self, user_query, manual_courses=None):
        if manual_courses is None: manual_courses = []
        
        parsed = parse_user_string(user_query, manual_courses)
        prompt = self.construct_prompt(parsed["instruction"], parsed["input"])
        
        # Embed only the instruction (text in "input" is too unorganized)
        raw_embed_response = self.llm.create_embedding(parsed["instruction"])

        if 'data' in raw_embed_response and len(raw_embed_response['data']) > 0:
            raw_list = raw_embed_response['data'][0]['embedding']
            
            embedding_np = np.array(raw_list)
            
            HIDDEN_SIZE = 8192
            
            if embedding_np.ndim == 1 and embedding_np.size > HIDDEN_SIZE:
                embedding_np = embedding_np.reshape(-1, HIDDEN_SIZE)

            pooled_embedding = np.mean(embedding_np, axis=0)
            
        else:
            pooled_embedding = np.zeros(8192)
        
        output = self.llm.create_completion(
            prompt,
            max_tokens=512,
            stop=["###", "</s>"],
            echo=False,
            temperature=0.7,
            logprobs=1 # Needed for calculating Z-axis
        )
        
        response_text = output['choices'][0]['text']
        
        # Take the average log-probability of the tokens generated
        # Higher (closer to 0) = more confidence in response. Lower (more negative) = less confidence in response
        # If viewing this within the rest of the project repository, view the file 'chat_history.txt' to view the exact values for each response from the model
        token_logprobs = output['choices'][0]['logprobs']['token_logprobs']
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
        avg_confidence = np.mean(valid_logprobs) if valid_logprobs else -999.0
        
        self.history_embeddings.append(pooled_embedding)
        self.history_confidences.append(avg_confidence)
        self.history_labels.append(user_query[:20] + "...")
        
        return response_text, avg_confidence

    def update_plot(self):
        if len(self.history_embeddings) < 3:
            print("[Graph] Need at least 3 queries to generate PCA graph.")
            return

        pca = PCA(n_components=2)
        coords = pca.fit_transform(self.history_embeddings)
        
        x_vals = coords[:, 0]
        y_vals = coords[:, 1]
        z_vals = self.history_confidences # prediction confidence

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        sc = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=60)
        
        # Attach the response text to the point on the graph
        for i, txt in enumerate(self.history_labels):
            ax.text(x_vals[i], y_vals[i], z_vals[i], txt, size=8)

        ax.set_xlabel('Input Variation 1')
        ax.set_ylabel('Input Variation 2')
        ax.set_zlabel('Model Prediction')
        ax.set_title(f'WPI Advisor Inference Landscape (n={len(self.history_embeddings)})')
        
        plt.savefig('live_inference_graph.png')
        print("Graph updated: 'live_inference_graph.png'")
        plt.close()

if __name__ == "__main__":
    advisor = AdvisorSystem(MODEL_PATH)
    
    print("\n--- WPI AI Advisor Ready! (Type 'quit' to exit) ---")
    
    while True:
        user_input = input("\nStudent: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        response, conf = advisor.get_advice(user_input, manual_courses=["CS4341"]) # hard-code courses to retrieve information in case 'input_parser.py' fails to
        
        print(f"\nModel Response: {response}")
        print(f"\n[System] Confidence Score: {conf:.4f}")
        
        advisor.update_plot()