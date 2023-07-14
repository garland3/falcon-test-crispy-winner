# %%
run_llm = True
if run_llm:
    from transformers import AutoTokenizer, pipeline, logging
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    import argparse

    model_name_or_path = "TheBloke/falcon-40b-instruct-GPTQ"
    # You could also download the model locally, and access it there
    # model_name_or_path = "/path/to/TheBloke_falcon-40b-instruct-GPTQ"

    model_basename = "gptq_model-4bit--1g"

    use_triton = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=use_triton,
            quantize_config=None)

    prompt = "Tell me about AI"
    prompt_template=f"""A helpful assistant who helps the user with any questions asked. 
    User: {prompt}
    Assistant:"""

    # print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    print(tokenizer.decode(output[0]))

    # Inference can also be done using transformers' pipeline
    # Note that if you use pipeline, you will see a spurious error message saying the model type is not supported
    # This can be ignored!  Or you can hide it with the following logging line:
    # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
    logging.set_verbosity(logging.CRITICAL)

    print("*** Pipeline:")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512*4,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    print(pipe(prompt_template)[0]['generated_text'])
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
            input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
            output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
            bot_message = tokenizer.decode(output[0])
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
