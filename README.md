# Transformer-Polyp-Tracker

This repository hosts the **Transformer-Polyp-Tracker**, a machine learning project that uses django, pytorch to accomplish the real-time detection of polyps in colocopic videos.

## Installation

To install the required packages, follow these steps:

1. Clone this repository to download the project files: `git clone https://github.com/alancarlosml/transformer-polyp-tracker.git`
2. Navigate to the project directory using the command line: `cd transformer-polyp-tracker`
3. Create a new virtual environment to keep the project's dependencies separate from other Python projects on your computer: `virtualenv venv` or `conda env create --name venv` (Conda)
4. Activate the virtual environment to use the packages installed in the environment: `source venv/bin/activate` (Linux) or `venv\Scripts\activate` (Windows). For Conda, `conda activate venv`
5. Install the required packages: `pip install -r requirements.txt`

## Usage

To run the project, follow these steps:

* Run the application: `python manager.py runserver`
* Open the browser: `http://127.0.0.1:8000/`

## Acknowledgements

This work was supported by Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES), Brazil - Finance Code 001, Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq), Brazil, and Fundação de Amparo à Pesquisa e ao Desenvolvimento Científico e Tecnológico do Maranhão (FAPEMA), Brazil.
