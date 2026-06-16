import gradio as gr

def hello():
    return "Working!"

demo = gr.Interface(fn=hello, inputs=[], outputs="text")

demo.launch()
