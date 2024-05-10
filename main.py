import gradio as gr
from utils import *  # import all functions from project utils
from saver import dATA

# Use block Class to build layouts flexibly
with gr.Blocks() as interface:
    with gr.Tab("Data Upload and Preporation"):  # with three main tabs module at first

        with gr.Row():  # first is an info textbox    
            output_data = gr.Textbox(label="Data infos",lines=10)

            # prepare a closure for btn
            enable = creat_btn("Confirm")

            with gr.Column(min_width=1):  # Using a dropdown to choose a datafile
                data_chose = gr.Dropdown(label="Choose a DataFile", choices=get_file_list())

                # Using a button to upload a datafile, invisible
                btn = gr.Button(variant='primary',value="Confirm", visible=False)

                # Listening dropdown to change btn component's attribute
                data_chose.change(fn=enable, inputs=data_chose, outputs=btn)
                btn.click(fn=upload_data, inputs=data_chose, outputs=output_data)
            
                # Using a slider to del missing values, output infos in `output_data`
                slide = gr.Slider(0, 1, 0.3, step=0.1, label="Missing ratio", info='Columns with a larger missing weight will be deleted', interactive=True)
                # Using a btn to confirm drop
                drop_btn = gr.Button(variant='primary', value="Drop!")
                normal_btn = gr.Button(variant='primary', value="Clear2full!")

            # The output box prints all string object columns
            output_string = gr.Textbox(label="Convert string columns",lines=10)
            # warning here for wrong btn click order
            drop_btn.click(fn=drop_missings, inputs=slide, outputs=[output_data, output_string])

            with gr.Column():  # Using a textbox to choose string columns
                data_choser = gr.Textbox(label="Choose some string columns")

                # Using checkboxs to Numeralization or del string columns
                is_log10 = gr.Checkbox(label="Enable log10", interactive=True)
                is_del = gr.Checkbox(label="Enable del", interactive=True)

                gr.Button(variant='primary', value="Numeralization!").click(inputs=[data_choser, is_log10, is_del], outputs=[output_data, output_string], fn=numeralization)
                with gr.Accordion("Examples", open=False):
                    gr.Examples(inputs=[data_choser, is_log10, is_del], 
                                examples=[
                        ['Id,Address,Summary,Heating,Cooling,Parking,Total spaces,Home type,Elementary School,High School,Heating features,Parking features,Parcel number,Zip',False, True],
                        ['Sold Price,Year built,Total interior livable area,Lot size,Tax assessed value,Annual tax amount,Listed Price', True, False],
                        ['Bedrooms, Bathrooms,Total spaces,Garage spaces,Elementary School Score,Elementary School Distance,High School Score,High School Distance', False, False]])

        with gr.Row():
            with gr.Accordion(open=False, label='Change range'):
                with gr.Column(scale=1, min_width=1):  # Frame the data range for certain columns
                    slider_min = gr.Slider(label="Min")
                    slider_max = gr.Slider(label="Max")

                    # a listener to update checkgroups as number columns
                    select_group = gr.CheckboxGroup(label="Choose columns",choices=[None])
                    output_string.change(fn=enable_check, inputs=select_group, outputs=select_group)
                    
                    change_btn = gr.Button(variant='primary', value="Change!")
                    change_btn.click(inputs=[slider_min, slider_max, select_group], fn=change_range, outputs=output_data)
            
            # 缺失值填充和标准化
            normal_btn.click(fn=clear2full, outputs=[output_data, output_string])

            with gr.Column(scale=5):  # frame head visualization
                slider_head = gr.Slider(label="Head counts", value=5, minimum=1, maximum=20, step=1)
                frame1 = gr.Dataframe(scale=5, label="Data Heads")

                # two listener to update the dataframe
                output_data.change(fn=dATA.show_head, outputs=frame1, inputs=slider_head)
                slider_head.change(fn=dATA.show_head, outputs=frame1, inputs=slider_head)
        
        # 还可以做一些列数据的自定义数据映射...
    

    with gr.Tab("Data Visualization"):
        with gr.Row():
            with gr.Column():
                with gr.Accordion(open=False, label="Choose columns"):  # chooser btn
                    map_choose = gr.CheckboxGroup(label="Choose columns",choices=[None])
                    output_string.change(fn=enable_check, inputs=map_choose, outputs=map_choose)

                # drawer btns
                scatters_btn = gr.Button(variant='primary', value="Make scatter!")
                hist_btn = gr.Button(variant='primary', value="Make hist!")
                normalize_button = gr.Button(variant='primary', value="Normalize!")
                button = gr.Button(variant='primary', value="Correlation matrix!")
                box_btn = gr.Button(variant='primary', value="Boxplot!")

            with gr.Column():  # 散点图
                scatter = gr.Plot(label='Scatter')
                scatters_btn.click(plot_scatter, inputs=map_choose, outputs=scatter)

        with gr.Row():  # 箱型图和直方图
            plot = gr.Plot(label='Hist')
            box = gr.Plot(label='BoxPlot')

            hist_btn.click(fn=plot_hist, inputs=map_choose, outputs=plot)
            box_btn.click(fn=plot_boxplots, inputs=map_choose, outputs=box)

        with gr.Row():  # 热力图/相关系数矩阵
            hot = gr.Plot()
            button.click(fn=plot_corr, outputs=hot)

        # normalize按钮触发器
        normalize_button.click(fn=normalize, inputs=map_choose, outputs=scatter)

    with gr.Tab("Model Training and analysis"):
        with gr.Row():  # print LOSS and chose model at first
            rmse = gr.Number(label="RMSLE", interactive=False)
            pre = gr.Radio(choices=['SalePrice', 'Sold Price'], label='the target')
            model_path = gr.Dropdown(label="Choose a model", choices=get_model_list())

        with gr.Row():  # choses for model trainning

            with gr.Column(scale=1):  # split or use new testdata
                # make closure for btn
                enable2 = creat_btn("Load Model!")
                with gr.Tab("Select old"):
                    show_size = gr.Textbox(label="Size infos", interactive=False)
                    # two split methods
                    methods = gr.Radio(label='methods', choices=['Split By Row number', 'Split By Column with date'])
                    
                    # a slider to choose a row split
                    slider1 = gr.Slider(visible=False)
                    # a number to choose a date split
                    date_choser = gr.Textbox(visible=False); col_choser = gr.Dropdown(visible=False)
                    
                    # listen to the radio
                    methods.change(fn=enable_more, inputs=methods, outputs=[slider1, date_choser, col_choser])
                    slider1.change(fn=split_data, inputs=slider1, outputs=show_size)
                    col_choser.change(fn=split_data, inputs=[date_choser, col_choser], outputs=show_size)

                    start_btn = gr.Button(variant='primary', value="Fit Model!")

                    # a load button, listened by the model_path
                    start_btn2 = gr.Button(variant='primary', value="Load Model!", visible=False)
                    model_path.change(fn=enable2, inputs=model_path, outputs=start_btn2)
                
                with gr.Tab("Select new"):
                    file_in = gr.File(label="Upload a TestData", type='filepath')
                    # a merge function!!!
                    file_in.change(fn=merge_new, inputs=file_in, outputs=[output_data, output_string])
                    
                    btn_start = gr.Button(variant='primary', value="Fit Model!")
                    btn_start2 = gr.Button(variant='primary', value="Load Model!", visible=False)
                    # for load model listener
                    model_path.change(fn=enable2, inputs=model_path, outputs=btn_start2)
            
            # main outputs infos with dataframe
            performance = gr.TextArea(label='Performance infos')
            perform_fig = gr.Plot(label='Performance plot')
        
        with gr.Row():  # output perforances frames
            perform_info = gr.Dataframe(label='Performance infos')
            importance = gr.Dataframe(label='Feature importance')

        # start trainning
        start_btn.click(fn=train_model, inputs=pre, outputs=[rmse, performance, perform_fig, perform_info, importance])
        start_btn2.click(fn=load_model, inputs=[pre, model_path], outputs=[rmse, performance, perform_fig, perform_info, importance])
        
        btn_start.click(fn=train_new, inputs=pre, outputs=[rmse, performance, perform_fig, perform_info, importance])
        btn_start2.click(fn=load_new, inputs=[pre, model_path], outputs=[rmse, performance, perform_fig, perform_info, importance])

interface.launch()