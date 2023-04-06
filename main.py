import subprocess
try:
    import PySimpleGUI as sg
except:
    subprocess.run(["pip", "install", 'pysimplegui'], check=True)
    import PySimpleGUI as sg
import os
import io
from PIL import Image

try:
    from kandinsky2 import get_kandinsky2
except:
    subprocess.run(["pip", "install", 'git+https://github.com/ai-forever/Kandinsky-2.git'], check=True)
    subprocess.run(["pip", "install", 'git+https://github.com/openai/CLIP.git'], check=True)
    from kandinsky2 import get_kandinsky2

def generate_thumbnail_grid(image_file_paths, num_cols=8):
    grid = []
    row = []
    for i, file_path in enumerate(image_file_paths):
        row.append(sg.Button('', key=f'thumbnail_{i}', image_filename='', size=(150, 150), pad=(5, 5), border_width=2))
        if (i + 1) % num_cols == 0:
            grid.append(row)
            row = []
    if row:
        grid.append(row)
    return grid


def main():
    image_file_paths = []

    sg.set_options(font=("Helvetica", 24))
    keys = ['input_text', 'num_steps', 'guidance_scale', 'h', 'w', 'prior_cf_scale', 'prior_steps', 'gui_scale']

    slider_labels = [
        {'text': 'Num steps', 'key': 'num_steps'},
        {'text': 'Guidance scale', 'key': 'guidance_scale'},
        {'text': 'Height', 'key': 'h'},
        {'text': 'Width', 'key': 'w'},
        {'text': 'Prior CF scale', 'key': 'prior_cf_scale'},
        {'text': 'Prior steps', 'key': 'prior_steps'}
    ]

    sliders = [[
        sg.Column([
            [sg.Text(size=(4, 1), key=f'value_{label["key"]}', justification='center')],
            [sg.Slider(range=(1, 100), default_value=75, orientation='v', key=label["key"], size=(8, 20),
                       enable_events=True) if label["key"] == 'num_steps' else
             sg.Slider(range=(1, 20), default_value=10, orientation='v', key=label["key"], size=(8, 20),
                       enable_events=True) if label["key"] == 'guidance_scale' else
             sg.Slider(range=(1, 1024), default_value=768, orientation='v', key=label["key"], size=(8, 20),
                       enable_events=True) if label["key"] in ['h', 'w'] else
             sg.Slider(range=(1, 10), default_value=4, orientation='v', key=label["key"], size=(8, 20),
                       enable_events=True)],
            [sg.Graph((50, 50), (0, 0), (50, 50), key=f"graph_{label['key']}")]
        ], element_justification='center', expand_x=True, expand_y=True)
        for label in slider_labels
    ]]
    layout = [
        [sg.Text('Enter text:', size=(50, 1)), sg.InputText(key='input_text')],
        [sg.Button('Initialize Model'),
         sg.Button('Generate Images'),],
        [
            sg.Column(
                [
                    [sg.Column(generate_thumbnail_grid(image_file_paths), scrollable=True, vertical_scroll_only=True,
                               size=(768, 200), key='thumbnail_grid', expand_x=True, expand_y=True)],
                    [sg.Column(
                        sliders,
                        element_justification='center', expand_x=True, expand_y=True
                    )],

                ],
                element_justification='center', expand_x=True, expand_y=True
            ),
            sg.VSeperator(),
            sg.Column(
                [
                    [sg.Image(key='image_display')],
                ],
                element_justification='center', expand_x=True, expand_y=True
            )
        ]
    ]

    window = sg.Window('Kandinsky2 Interface', layout, resizable=True, finalize=True)
    for label in slider_labels:
        window[f"graph_{label['key']}"].draw_text(label['text'], (25, 25), angle=45, font=("Helvetica", 10))
    model = None
    images = None

    # Create the 'outputs' folder if it doesn't exist
    os.makedirs('outputs', exist_ok=True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == 'Initialize Model':
            model = get_kandinsky2('cuda', task_type='text2img')
            sg.popup('Model initialized')
        elif event == 'Generate Images' and model is not None:
            if values['input_text'] == "":
                values['input_text'] = "Text saying empty"
            try:
                images = model.generate_text2img(
                    values['input_text'],
                    num_steps=int(values['num_steps']),
                    batch_size=1,
                    guidance_scale=int(values['guidance_scale']),
                    h=int(values['h']),
                    w=int(values['w']),
                    sampler='p_sampler',
                    prior_cf_scale=int(values['prior_cf_scale']),
                    prior_steps=str(int(values['prior_steps']))
                )
            except Exception as e:
                print(e)
            image_bytes = io.BytesIO()
            images[0].save(image_bytes, format='PNG')
            window['image_display'].update(data=image_bytes.getvalue())

            for i, image in enumerate(images[:4]):
                file_path = os.path.join('outputs', f'image_{len(image_file_paths) + i}.png')
                image.save(file_path)
                image_file_paths.append(file_path)

                thumbnail_bytes = io.BytesIO()
                image.thumbnail((150, 150))
                image.save(thumbnail_bytes, format='PNG')

                thumbnail_key = f'thumbnail_{len(image_file_paths) - 1}'
                button = sg.Button('', key=thumbnail_key, image_data=thumbnail_bytes.getvalue(), size=(150, 150),
                                   pad=(5, 5), border_width=2)
                window.extend_layout(window['thumbnail_grid'], [[button]])
        elif event == 'Save Images' and images is not None:
            save_path = sg.popup_get_folder('Select a folder to save the images')
            if save_path:
                for i, image in enumerate(images):
                    image.save(os.path.join(save_path, f'image_{i}.png'))
                sg.popup('Images saved')
        elif event.startswith('thumbnail_') and image_file_paths is not None:
            thumbnail_index = int(event.split('_')[-1])
            if 0 <= thumbnail_index < len(image_file_paths):
                image = Image.open(image_file_paths[thumbnail_index])
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='PNG')
                window['image_display'].update(data=image_bytes.getvalue())
    window.close()

main()
