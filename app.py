import gradio as gr
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from modeling_florence2 import Florence2ForConditionalGeneration
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon  
import numpy as np
import random
import json


with open("config.json", "r") as f:
    config = json.load(f)

d_model = config['text_config']['d_model']
num_layers = config['text_config']['encoder_layers']
attention_heads = config['text_config']['encoder_attention_heads']
vocab_size = config['text_config']['vocab_size']
max_length = config['text_config']['max_length']
beam_size = config['text_config']['num_beams']
dropout = config['text_config']['dropout']
activation_function = config['text_config']['activation_function']
no_repeat_ngram_size = config['text_config']['no_repeat_ngram_size']
patch_size = config['vision_config']['patch_size'][0]
temporal_embeddings = config['vision_config']['visual_temporal_embedding']['max_temporal_embeddings']

title = """# 🙋🏻‍♂️Bem-vindo ao ÓUSI PREMIUM/florence"""
description = """
Este aplicativo apresenta o modelo **ÓUSI PREMIUM/florence**, um poderoso sistema de IA projetado para tarefas de **geração de texto e imagem**. O modelo é capaz de lidar com tarefas complexas como detecção de objetos, legendagem de imagens, OCR (Reconhecimento Óptico de Caracteres) e análise detalhada de imagens baseadas em regiões.

### Uso e Flexibilidade do Modelo

- **Sem Repetição de N-Gramas**: Para reduzir a repetição na geração de texto, o modelo é configurado com um **no_repeat_ngram_size** de **{no_repeat_ngram_size}**, garantindo saídas mais diversificadas e significativas.
- **Estratégias de Amostragem**: ÓUSI PREMIUM/florence oferece estratégias de amostragem flexíveis, incluindo **top-k** e **top-p (nucleus) sampling**, permitindo tanto geração criativa quanto restrita, com base nas necessidades do usuário.

📸📈✍🏻florence é um modelo robusto capaz de lidar com várias tarefas de **texto e imagem** com alta precisão e flexibilidade, tornando-se uma ferramenta valiosa para pesquisas acadêmicas e aplicações práticas.

### **Como Usar**:
1. **Faça o Upload de uma Imagem**: Selecione uma imagem para processamento.
2. **Escolha uma Tarefa**: Escolha uma tarefa no menu suspenso, como "Legenda", "Detecção de Objetos", "OCR", etc.
3. **Processar**: Clique no botão "Processar" para permitir que ÓUSI PREMIUM/florence analise a imagem e gere a saída.
4. **Ver Resultados**: Dependendo da tarefa, você verá uma imagem processada (por exemplo, com caixas delimitadoras ou rótulos) ou um resultado baseado em texto (por exemplo, uma legenda gerada ou texto extraído).

Você pode redefinir a interface a qualquer momento clicando no botão **Redefinir**.

### **Tarefas Disponíveis**:
- **✍🏻Legenda**: Gere uma descrição concisa da imagem.
- **📸Detecção de Objetos**: Identifique e rotule objetos dentro da imagem.
- **📸✍🏻OCR**: Extraia texto da imagem.
- **📸Proposta de Região**: Detecte regiões-chave na imagem para legendagem detalhada.
"""

model_presentation = f"""
O modelo **ÓUSI PREMIUM/florence** é um modelo de ponta para tarefas de geração condicional, projetado para ser altamente eficaz em tarefas de **texto** e **visão**. É construído como uma arquitetura de **codificador-decodificador**, que permite maior flexibilidade e desempenho na geração de saídas com base em entradas diversificadas.

### Principais Características

- **Arquitetura do Modelo**: ÓUSI PREMIUM/florence usa uma estrutura de codificador-decodificador, o que o torna eficaz em tarefas como **geração de texto**, **resumo** e **tradução**. Ele possui **{num_layers} camadas** tanto para o codificador quanto para o decodificador, com uma dimensão do modelo (`d_model`) de **{d_model}**.
- **Geração Condicional**: O modelo pode gerar texto condicionalmente, com um comprimento máximo de **{max_length} tokens** para cada sequência gerada, tornando-o ideal para tarefas que exigem saída concisa.
- **Busca em Feixe**: ÓUSI PREMIUM/florence suporta **busca em feixe** com até **{beam_size} feixes**, permitindo geração de texto mais diversa e precisa explorando múltiplas potenciais saídas antes de selecionar a melhor.
- **Tokenização**: Inclui um tokenizador com um vocabulário de **{vocab_size} tokens**. Tokens especiais como **bos_token_id (0)** e **eos_token_id (2)** ajudam a controlar o processo de geração, marcando o início e o fim de uma sequência.
- **Mecanismo de Atenção**: Tanto o codificador quanto o decodificador utilizam **{attention_heads} cabeças de atenção** por camada, garantindo que o modelo possa focar em partes relevantes da entrada ao gerar texto.
- **Dropout e Ativação**: ÓUSI PREMIUM/florence emprega uma **função de ativação {activation_function}** e uma **taxa de dropout de {dropout}**, o que melhora o desempenho do modelo prevenindo overfitting e melhorando a generalização.
- **Configuração de Treinamento**: O modelo usa precisão **float32** para treinamento e suporta fine-tuning para tarefas específicas ao configurar `finetuning_task` apropriadamente.

### Integração de Visão

Além das tarefas de texto, ÓUSI PREMIUM/florence também incorpora **capacidades de visão**:
- **Processamento de Imagem Baseado em Patches**: O componente de visão opera em patches de imagem com um tamanho de patch de **{patch_size}x{patch_size}**.
- **Embedding Temporal**: Tarefas visuais se beneficiam de embeddings temporais com até **{temporal_embeddings} passos**, tornando o florence bem adequado para análise de vídeo.
"""

joinus = """ÓUSI PREMIUM/florence é um modelo de IA de ponta que oferece uma ampla gama de recursos para tarefas de texto e visão. Se você deseja colaborar, contribuir ou saber mais sobre o projeto, sinta-se à vontade para entrar em contato conosco! Junte-se a nós para explorar o potencial da IA e criar soluções inovadoras para o futuro.
"""
how_to_use = """As configurações avançadas permitem que você ajuste o processo de geração de texto. Aqui está o que cada configuração faz e como usá-la:

### Top-k (Padrão: 50)
A amostragem top-k limita a seleção do próximo token aos k tokens mais prováveis.

- **Valores mais baixos** (por exemplo, 10) tornam a saída mais focada e determinística.
- **Valores mais altos** (por exemplo, 100) permitem saídas mais diversificadas.

**Exemplo:** Para uma tarefa de escrita criativa, tente definir top-k para 80 para uma linguagem mais variada.

### Top-p (Padrão: 1.0)
A amostragem top-p (ou nucleus) seleciona do menor conjunto de tokens cuja probabilidade cumulativa excede p.

- **Valores mais baixos** (por exemplo, 0.5) tornam a saída mais focada e coerente.
- **Valores mais altos** (por exemplo, 0.9) permitem saídas mais diversificadas e potencialmente criativas.

**Exemplo:** Para uma legenda factual, defina top-p para 0.7 para equilibrar precisão e criatividade.

### Penalidade de Repetição (Padrão: 1.0)
Esta penaliza a repetição no texto gerado.

- **Valores próximos a 1.0** têm efeito mínimo na repetição.
- **Valores mais altos** (por exemplo, 1.5) desencorajam mais fortemente a repetição.

**Exemplo:** Se você notar frases repetidas, tente aumentar para 1.2 para um texto mais variado.

### Número de Feixes (Padrão: 3)
A busca em feixe explora múltiplas sequências possíveis em paralelo.

- **Valores mais altos** (por exemplo, 5) podem levar a melhor qualidade, mas geração mais lenta.
- **Valores mais baixos** (por exemplo, 1) são mais rápidos, mas podem produzir resultados de menor qualidade.

**Exemplo:** Para tarefas complexas como legendagem densa, tente aumentar para 5 feixes.

### Máximo de Tokens (Padrão: 512)
Define o comprimento máximo do texto gerado.

- **Valores mais baixos** (por exemplo, 100) para saídas concisas.
- **Valores mais altos** (por exemplo, 1000) para descrições mais detalhadas.

**Exemplo:** Para uma descrição detalhada da imagem, defina o máximo de tokens para 800 para uma saída abrangente.

Lembre-se, essas configurações interagem entre si, então experimentar diferentes combinações pode levar a resultados interessantes!
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = Florence2ForConditionalGeneration.from_pretrained("PleIAs/Florence-PDF", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("PleIAs/Florence-PDF", trust_remote_code=True)

TASK_PROMPTS = {
    "✍🏻Caption": "<CAPTION>",
    "✍🏻✍🏻Caption": "<DETAILED_CAPTION>",
    "✍🏻✍🏻✍🏻Caption": "<MORE_DETAILED_CAPTION>",
    "📸Object Detection": "<OD>",
    "📸Dense Region Caption": "<DENSE_REGION_CAPTION>",
    "📸✍🏻OCR": "<OCR>",
    "📸✍🏻OCR with Region": "<OCR_WITH_REGION>",
    "📸Region Proposal": "<REGION_PROPOSAL>",
    "📸✍🏻Object Detection with Description": "<OD>",  # Start with Object Detection
    # We will handle the detailed description separately in the code
}

# Update IMAGE_TASKS and TEXT_TASKS
IMAGE_TASKS = ["📸Object Detection", "📸Dense Region Caption", "📸Region Proposal", "📸✍🏻OCR with Region", "📸✍🏻Object Detection with Description"]
TEXT_TASKS = ["✍🏻Caption", "✍🏻✍🏻Caption", "✍🏻✍🏻✍🏻Caption", "📸✍🏻OCR", "📸✍🏻OCR with Region", "📸✍🏻Object Detection with Description"]

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def plot_bbox(image, data, use_quad_boxes=False):
    fig, ax = plt.subplots()
    ax.imshow(image)

    if use_quad_boxes:
        for quad_box, label in zip(data.get('quad_boxes', []), data.get('labels', [])):
            quad_box = np.array(quad_box).reshape(-1, 2)
            poly = Polygon(quad_box, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(poly)
            plt.text(quad_box[0][0], quad_box[0][1], label, color='white', fontsize=8,
                     bbox=dict(facecolor='red', alpha=0.5))
    else:
        bboxes = data.get('bboxes', [])
        labels = data.get('labels', [])
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')

    return fig

def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                  "{}".format(label),
                  align="right",
                  fill=color)
        
    return image

def draw_bounding_boxes(image, quad_boxes, labels, color=(0, 255, 0), thickness=2):
    """
    Draws quadrilateral bounding boxes on the image.
    """
    for i, quad in enumerate(quad_boxes):
        points = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))  # Reshape the quad points for drawing
        image = cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        label_pos = (int(quad[0]), int(quad[1]) - 10)  
        cv2.putText(image, labels[i], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return image

def process_image(image, task):
    prompt = TASK_PROMPTS[task]
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

    return parsed_answer


def main_process(image, task, top_k, top_p, repetition_penalty, num_beams, max_tokens):
    prompt = TASK_PROMPTS[task]
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=num_beams,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
    return parsed_answer

def process_and_update(image, task, top_k, top_p, repetition_penalty, num_beams, max_tokens):
    if image is None:
        return None, gr.update(visible=False), "Please upload an image first.", gr.update(visible=True)
    
    if task == "📸✍🏻Object Detection with Description":
        # Perform Object Detection first
        od_prompt = TASK_PROMPTS["📸Object Detection"]
        od_inputs = processor(text=od_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        od_generated_ids = model.generate(
            **od_inputs,
            max_new_tokens=max_tokens,
            num_beams=num_beams,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        od_generated_text = processor.batch_decode(od_generated_ids, skip_special_tokens=False)[0]
        od_parsed_answer = processor.post_process_generation(od_generated_text, task=od_prompt, image_size=(image.width, image.height))
        
        # Display Bounding Boxes
        fig = plot_bbox(image, od_parsed_answer.get('<OD>', {}))
        output_image = fig_to_pil(fig)
        
        # Then perform Detailed Description
        dd_prompt = TASK_PROMPTS["✍🏻✍🏻✍🏻Caption"]
        dd_inputs = processor(text=dd_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        dd_generated_ids = model.generate(
            **dd_inputs,
            max_new_tokens=max_tokens,
            num_beams=num_beams,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        dd_generated_text = processor.batch_decode(dd_generated_ids, skip_special_tokens=False)[0]
        dd_parsed_answer = processor.post_process_generation(dd_generated_text, task=dd_prompt, image_size=(image.width, image.height))
        text_output = str(dd_parsed_answer)
        
        return output_image, gr.update(visible=True), text_output, gr.update(visible=True)
    else:
        # Existing processing for other tasks
        result = main_process(image, task, top_k, top_p, repetition_penalty, num_beams, max_tokens)
        
        if task in IMAGE_TASKS:
            if task == "📸✍🏻OCR with Region":
                fig = plot_bbox(image, result.get('<OCR_WITH_REGION>', {}), use_quad_boxes=True)
                output_image = fig_to_pil(fig)
                text_output = result.get('<OCR_WITH_REGION>', {}).get('recognized_text', 'No text found')
                return output_image, gr.update(visible=True), text_output, gr.update(visible=True)
            else:
                fig = plot_bbox(image, result.get(TASK_PROMPTS[task], {}))
                output_image = fig_to_pil(fig)
                return output_image, gr.update(visible=True), None, gr.update(visible=False)
        else:
            return None, gr.update(visible=False), str(result), gr.update(visible=True)

def reset_outputs():
    return None, gr.update(visible=False), None, gr.update(visible=True)

with gr.Blocks(title="Tonic's 🙏🏻PLeIAs/📸📈✍🏻Florence-PDF") as iface:
    with gr.Column():
        with gr.Row():
            gr.Markdown(title)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():                    
                    gr.Markdown(model_presentation)
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(description)
        with gr.Row():
            with gr.Accordion("🫱🏻‍🫲🏻Join Us", open=True):
                gr.Markdown(joinus)
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Input Image")
                task_dropdown = gr.Dropdown(list(TASK_PROMPTS.keys()), label="Task", value="✍🏻Caption")            
                with gr.Row():
                    submit_button = gr.Button("📸📈✍🏻Process")
                    reset_button = gr.Button("♻️Reset")
                with gr.Accordion("🧪Advanced Settings", open=False):
                    with gr.Accordion("🏗️How To Use", open=True):
                        gr.Markdown(how_to_use)                        
                    top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="Top-p")
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.0, step=0.01, label="Repetition Penalty")
                    num_beams = gr.Slider(minimum=1, maximum=6, value=3, step=1, label="Number of Beams")
                    max_tokens = gr.Slider(minimum=1, maximum=1024, value=1000, step=1, label="Max Tokens")
            with gr.Column(scale=1):    
                output_image = gr.Image(label="ÓUSI PREMIUM/florence", visible=False)
                output_text = gr.Textbox(label="ÓUSI PREMIUM/florence", visible=False)
    
    submit_button.click(
        fn=process_and_update,
        inputs=[image_input, task_dropdown, top_k, top_p, repetition_penalty, num_beams, max_tokens],
        outputs=[output_image, output_image, output_text, output_text]
    )
    
    reset_button.click(
        fn=reset_outputs,
        inputs=[],
        outputs=[output_image, output_image, output_text, output_text]
    )
    
    task_dropdown.change(
        fn=lambda task: (gr.update(visible=task in IMAGE_TASKS), gr.update(visible=task in TEXT_TASKS)),
        inputs=[task_dropdown],
        outputs=[output_image, output_text]
    )

iface.launch()