# Requires: pip install PySimpleGUI
import PySimpleGUI as sg
import subprocess
import sys
import os

# Layout for the GUI
layout = [
    [sg.Text('Peptide Source:'), sg.Combo(['random', 'fasta', 'llm'], default_value='random', key='-SOURCE-', enable_events=True)],
    [sg.Text('LLM Model:'), sg.Combo(['protgpt2', 'esm2'], default_value='protgpt2', key='-LLM_MODEL-', enable_events=True)],
    [sg.Text('Peptide Length:'), sg.Input(key='-LENGTH-', size=(5,1)), sg.Text('Count:'), sg.Input(key='-COUNT-', size=(5,1))],
    [sg.Text('FASTA File:'), sg.Input(key='-FASTA-', size=(30,1)), sg.FileBrowse(file_types=(('FASTA Files', '*.fasta *.faa'),), key='-FASTA_BROWSE-')],
    [sg.Text('Output File:'), sg.Input(key='-OUTPUT-', size=(30,1)), sg.FileSaveAs(file_types=(('FASTA Files', '*.fasta'),), key='-OUTPUT_BROWSE-')],
    [sg.Frame('LLM Options (optional)', [
        [sg.Text('Temperature:'), sg.Input(key='-TEMP-', size=(5,1)),
         sg.Text('Top-k:'), sg.Input(key='-TOPK-', size=(5,1)),
         sg.Text('Top-p:'), sg.Input(key='-TOPP-', size=(5,1)),
         sg.Text('Repetition Penalty:'), sg.Input(key='-REPPEN-', size=(5,1))]
    ])],
    [sg.Button('Generate'), sg.Exit()],
    [sg.Multiline(size=(80,10), key='-OUTPUTBOX-', autoscroll=True, disabled=True)]
]

window = sg.Window('Peptide Control Generator', layout)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break

    # Enable/disable inputs based on source
    if event == '-SOURCE-':
        if values['-SOURCE-'] == 'fasta':
            window['-FASTA-'].update(disabled=False)
            window['-FASTA_BROWSE-'].update(disabled=False)
            window['-LLM_MODEL-'].update(disabled=True)
        elif values['-SOURCE-'] == 'llm':
            window['-FASTA-'].update(disabled=True)
            window['-FASTA_BROWSE-'].update(disabled=True)
            window['-LLM_MODEL-'].update(disabled=False)
        else:
            window['-FASTA-'].update(disabled=True)
            window['-FASTA_BROWSE-'].update(disabled=True)
            window['-LLM_MODEL-'].update(disabled=True)

    if event == 'Generate':
        source = values['-SOURCE-']
        length = values['-LENGTH-']
        count = values['-COUNT-']
        fasta_file = values['-FASTA-']
        output_file = values['-OUTPUT-']
        llm_model = values['-LLM_MODEL-']
        temp = values['-TEMP-']
        topk = values['-TOPK-']
        topp = values['-TOPP-']
        reppen = values['-REPPEN-']

        # Build command
        cmd = [sys.executable, 'generate_control_peptides.py', '--length', length, '--count', count, '--source', source, '--output', output_file]
        if source == 'fasta' and fasta_file:
            cmd += ['--fasta_file', fasta_file]
        if source == 'llm':
            cmd += ['--llm_model', llm_model]
            if temp:
                cmd += ['--temperature', temp]
            if topk:
                cmd += ['--top_k', topk]
            if topp:
                cmd += ['--top_p', topp]
            if reppen:
                cmd += ['--repetition_penalty', reppen]

        outputbox = window['-OUTPUTBOX-']
        if outputbox is not None:
            outputbox.update('Running command:\n' + ' '.join(cmd) + '\n')
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if proc.stdout is not None:
                for line in proc.stdout:
                    if outputbox is not None:
                        outputbox.update(line, append=True)
            proc.wait()
            if outputbox is not None:
                outputbox.update(f'\nDone. Exit code: {proc.returncode}\n', append=True)
        except Exception as e:
            if outputbox is not None:
                outputbox.update(f'Error: {e}\n', append=True)

window.close() 