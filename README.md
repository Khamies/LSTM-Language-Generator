# LSTM-Language-Generator
A Pytorch LSTM Language model trained on Penn Tree Bank Dataset.

![](/home/waleed/Desktop/papers/LSTM-Variational-AutoEncoder/media/model.jpg)

### Table of Contents

- **[Introduction](#Introduction)**
- **[Setup](#Setup)**
- [**Run the code**](#Run-the-code)
- **[Training](#Training)**
- **[Inference](#Inference)**
- **[Play with the model](#Play-with-the-model)**
- **[Connect with me](#Connect-with-me)**
- **[License](#License)** 



### Introduction

This is a PyTorch Implementation for an LSTM-based language model. The model is trained on Penn Tree Bank dataset using Adam optimizer with a learning rate `0.001` and 10 epochs. 

### Setup

The code is using `pipenv` as a virtual environment and package manager. To run the code, all you need is to install the necessary dependencies. open the terminal and type:

- `git clone https://github.com/Khamies/LSTM-Language-Generator.git` 
- `cd LSTM-Language-Generator`
- `pipenv install`

And you should be ready to go to play with code and build upon it!

### Run the code

- To train the model, run: `python main.py`

- To train the model with specific arguments, run: `python main.py --batch_size=64`. The following command-line arguments are available:
  - Batch size: `--batch_size`
  - bptt: `--bptt`
  - Learning rate: `--lr`
  - Embedding size: `--embed_size`
  - Hidden size: `--hidden_size`
  - Latent size: `--latent_size`

### Training

The model is trained on 10 epochs using Adam as an optimizer with a learning rate 0.001. Here are the results from training the model:

- **Negative Likelihood Loss**

  <img src="./media/nll_loss.jpg" align="center" height="300" width="500" >

### Sample Generation

Here are some generated samples from the model:

```markdown
he said <pad> is n't expected to be the first quarter of
the company said <eos> in the u.s and japan
```

## Play with the model

To play with the model, a jupyter notebook has been provided, you can find it [here](https://github.com/Khamies/LSTM-Sequence-VAE/blob/main/play_with_model.ipynb)

### Citation

> ```
> @misc{Khamies2021SequenceVAE,
> author = {Khamies, Waleed},
> title = {PyTorch Implementation of Generating Sentences from a Continuous Space by Bowman et al. 2015},
> year = {2021},
> publisher = {GitHub},
> journal = {GitHub repository},
> howpublished = {\url{https://github.com/Khamies/Sequence-VAE}},
> }
> ```

### Connect with me :slightly_smiling_face:

For any question or a collaboration, drop me a message [here](mailto:khamiesw@outlook.com?subject=[GitHub]%20Sequence-VAE%20Repo)

Follow me on [Linkedin](https://www.linkedin.com/in/khamiesw/)!

**Thank you :heart:**

### License 

![](https://img.shields.io/github/license/khamies/LSTM-Language-Generator)

