from pathlib import Path
import gradio as gr
from saver import dATA


def upload_data(file_opt):
    
    file = Path('./data/' + file_opt)
    if file.suffix == '.csv':
        dATA.load_csv(file=file)
    elif file.suffix == '.xlsx':
        dATA.load_xls(file=file)
    else:
        dATA.load_feather(file=file)
    return dATA.info
    

def get_file_list():
    try:
        thelist = [f.stem + f.suffix for f in Path('./data').glob('*') if f.is_file()]
        if thelist is not []:
            return thelist
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Check your working directory")
        exit(1)

def get_model_list():
    try:
        thelist = [f for f in Path('./AutogluonModels').glob('*') if f.is_dir()]
        return thelist
    except FileNotFoundError:
        print("Check your working directory")
        exit(1)

def creat_btn(string):
    def enable_btn(chose):
        if chose is not None:
            return gr.Button(variant='primary',value=string, visible=True)
    return enable_btn

def enable_more(methods_value):
    if methods_value == 'Split By Row number':
        return [
            gr.Slider(label='Row number position to split', minimum=1000, maximum=100000, step=1000, value=10000, interactive=True, visible=True),
            gr.Textbox(visible=False),
            gr.Dropdown(visible=False)
        ]
    elif methods_value == 'Split By Column with date':
        return [
            gr.Slider(visible=False),
            gr.Textbox(interactive=True, label="Input a date", visible=True, placeholder="2022-01-01"),
            gr.Dropdown(interactive=True, label='Choose a date column', choices=dATA.data.columns.tolist(), visible=True)
        ]

def enable_check(input):
    if dATA.data is not None:
        output = dATA.data.select_dtypes(include=['number']).columns.tolist()
        return gr.CheckboxGroup(interactive=True, choices=output, label="Choose columns")
    else:
        return input