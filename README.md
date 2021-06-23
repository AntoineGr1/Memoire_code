# Repository for the Master's Thesis

# INSTALLATION
Libraries.txt contains the list of all libraries neccessary for the reproductibility of the work

- For install the libraries with pip : pip install -r libraries.txt

- For anaconda : 
    - conda create -y --name your_env python=3.9
    - conda install --force-reinstall -y -q --name your_env -c conda-forge --file libraries.txt


# STRUCTURE
## Directory
- architecture_base: architecture of baseline
- architecture_img: generated image file
- architecture_json: generated JSON file
- architecture_log: genereted log file
- architecture_py: genereted python file

## File

- Generateur_Architecture_Json*: notebook that generate architecture and convert to JSON file
- Compilateur*: notebook that convert JSON file to Python file
- Analyse_architecture*: notebook with analysis of results
- architecture_results*.csv: csv file that containts information of the result of the architecture (accuracy, training, time,...)

# EXECUTION 

- Run first one of the Generateur_Architecture_Json file
- Then run the conresponding compilateur file
- Then execute the generated python file
- Then see the result in the corresponding Analyse_architecture file