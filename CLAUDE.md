# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bioinformatics tool for generating control peptides used in neoantigen analysis and benchmarking with pVACtools. The project targets cancer immunotherapy research by providing high-quality reference peptide sets.

## Core Architecture

The project consists of two main components:

1. **CLI Tool** (`generate_control_peptides.py`) - Main peptide generation engine
2. **GUI Interface** (`peptide_gui.py`) - User-friendly wrapper using PySimpleGUI

### Peptide Generation Methods

Three generation strategies are implemented as separate functions:

- `generate_random_peptides()` - Random amino acid sequences from 20 standard amino acids
- `sample_peptides_from_fasta()` - Extracts subsequences from existing protein sequences 
- `generate_llm_peptides()` - Uses HuggingFace protein language models (ProtGPT2 or ESM-2) for biologically plausible sequences

### Key Design Patterns

- **Input validation**: Comprehensive parameter bounds checking (length 1-50, count 1-10M)
- **Error handling**: Graceful degradation with informative error messages
- **Reproducibility**: Random seed support for deterministic results
- **Batch processing**: Optimized generation for large peptide counts

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Testing the CLI
```bash
# Test random generation
python generate_control_peptides.py --length 9 --count 10 --source random --output test.fasta --seed 42

# Test FASTA sampling (requires protein sequences in data/)
python generate_control_peptides.py --length 9 --count 10 --source fasta --fasta_file data/GCF_000001405.40/protein.faa --output test.fasta

# Test ProtGPT2 generation (good for short peptides, 8-12 AA)
python generate_control_peptides.py --length 9 --count 5 --source llm --llm_model protgpt2 --temperature 1.2 --output test.fasta

# Test ESM-2 generation (better for longer peptides, 10+ AA)
python generate_control_peptides.py --length 15 --count 5 --source llm --llm_model esm2 --temperature 1.0 --output test.fasta
```

### Testing the GUI
```bash
python peptide_gui.py
```

### Code Validation
```bash
# Check syntax
python3 -m py_compile generate_control_peptides.py
python3 -m py_compile peptide_gui.py

# View help
python generate_control_peptides.py --help
```

## Important Implementation Details

### Multi-LLM Architecture
- **ProtGPT2**: 738M parameter model, token-based (~4 AA per token), good for short peptides (8-12 AA)
- **ESM-2**: 650M parameter model, amino acid-level tokenization, better for longer peptides (10+ AA)
- Modular design allows easy addition of new protein language models
- Uses batch generation for efficiency (ProtGPT2: 50/batch, ESM-2: 10/batch)
- Includes retry logic with model-specific attempt limits
- Filters generated text to valid amino acids only

### GUI-CLI Communication
The GUI builds command-line arguments and executes the CLI tool via subprocess, streaming output in real-time to the GUI text box.

### Input Validation Bounds
- Peptide length: 1-50 amino acids
- Count: 1-10,000,000 peptides  
- LLM parameters: temperature > 0, top_k ≥ 1, top_p (0,1], repetition_penalty > 0
- Model selection: protgpt2 (recommended for ≤12 AA), esm2 (recommended for ≥10 AA)

### File Formats
- Input: FASTA format (`.fasta`, `.faa`) for protein sequences
- Output: FASTA format with headers like `>peptide_1`, `>peptide_2`, etc.

## Dependencies

Core dependencies are specified in `requirements.txt`:
- PySimpleGUI (GUI framework)
- transformers (protein language models)
- torch (PyTorch backend)
- sentencepiece (tokenization support)

The project uses only Python standard library beyond these requirements.