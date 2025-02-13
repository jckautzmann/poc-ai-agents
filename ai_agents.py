from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load pre-trained models (replace with your preferred models)
model_name_or_path = "google/flan-t5-base"  # Example: A sequence-to-sequence model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

# Define AI agents
class Agent:
    def __init__(self, name, model, tokenizer):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []

    def generate_response(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.conversation_history.append((input_text, response))
        return response

# Create AI agents
agent1 = Agent("Agent 1", model, tokenizer)
agent2 = Agent("Agent 2", model, tokenizer)

# Example conversation
agent1_question = "What is the capital of France?"
agent1_response = agent1.generate_response(agent1_question)
print(f"{agent1.name}: {agent1_question}")
print(f"{agent1.name}: {agent1_response}")

agent2_question = "What is the largest planet in our solar system?"
agent2_response = agent2.generate_response(agent2_question)
print(f"{agent2.name}: {agent2_question}")
print(f"{agent2.name}: {agent2_response}")