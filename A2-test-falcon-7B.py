# %%
run_llm = True
if run_llm:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    import torch

    model = "tiiuae/falcon-7b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    sequences = pipeline(
    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

# %%


import gradio as gr
import random
import time

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        # prompt = "Tell me about AI"
        prompt_template=f"""A helpful assistant who helps the user with any questions asked. 
User: {message}
Assistant:"""
        if run_llm:
            sequences = pipeline(prompt_template,
                max_length=200,
                do_sample=True,
                top_k=1,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
            bot_message = sequences[0]['generated_text']
        else:
            bot_message = "I'm a chatbot"    
        
        chat_history.append((message, bot_message))
        # time.sleep(2)
        return "", chat_history



    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    def clear_chat(chatbot):
        chatbot = []
        # gr.update([chatbot])
        return  (chatbot)
    clear.click(clear_chat, [chatbot], [chatbot])

demo.launch()
