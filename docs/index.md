## Course Overview

Welcome to the Machine Learning and Deep Learning for Biomolecular Systems and Material Science course. This 
program will equip you with the knowledge and skills to apply cutting-edge ML techniques to molecular design, 
property prediction, and materials discovery.

## Prerequisites

One can work with Python workflows in different manners, one of them is by running Python scripts on the 
terminal (not allowed on login nodes) ``python script.py``, but here one will need to change back and forth from 
the command line to a text editor if the script needs to be modified. 

A more interactive manner to work with Python workflows is by using Jupyter notebooks which are supported 
on Kebnekaise. You can login [HPC2N Portal](https://docs.hpc2n.umu.se/tutorials/connections/#open__ondemand){:target="_blank"}
using your HPC2N credentials. Then you can start a 
[Jupyter interactive App](https://docs.hpc2n.umu.se/tutorials/connections/#interactive__apps__-__jupyter__notebook){:target="_blank"} 
where you can choose either a CPU node (``any``) or a GPU node (``any_gpu``) to run your code. 

When you enter the Jupyter session, you can start new terminals or new Jupyter noteboosk.

### Example for the Day1 

On a terminal in the Jupyter interactive App, execute the following commands (copy and paste):

```bash
ml GCCcore/13.2.0 JupyterLab/4.2.0  CUDA/12.6.0                                                                                               
                                                                                                                                              
python -m venv day1                                                                                                                           
source day1/bin/activate                                                                                                                      
                                                                                                                                              
pip install --no-cache-dir --no-build-isolation ipykernel                                                                                     
pip install --no-cache-dir scikit-learn numpy                                                                                                 
pip install --no-cache-dir matplotlib                                                                                                         
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126                                    
python -m ipykernel install --user --name=day1
```

Then, create a new Jupyter notebook selecting the kernel ``day1`` that you recently created. Test that 
the kernel is working by executing 

```python
print("Hello world")
```

in your notebook. 


