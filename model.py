import torch

from data.ptb import PTB


class LSTM_Language(torch.nn.Module):

  def __init__(self, vocab_size, embed_size, hidden_size, latent_size, num_layers=1):
    super(LSTM_Language, self).__init__()

    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Variables
    self.num_layers = num_layers
    self.lstm_factor = num_layers
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.latent_size = latent_size

    # For dictionary lookups 
    self.dictionary = PTB(data_dir="./data", split="train", create_data= False, max_sequence_length= 60)
  
    # X: bsz * seq_len * vocab_size 
    # Embedding
    self.embed = torch.nn.Embedding(num_embeddings= self.vocab_size,embedding_dim= self.embed_size)

    #    X: bsz * seq_len * vocab_size 
    #    X: bsz * seq_len * embed_size

    # Encoder Part
    self.lstm = torch.nn.LSTM(input_size= self.embed_size,hidden_size= self.hidden_size, batch_first=True, num_layers= self.num_layers)
    self.output = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.vocab_size)
    self.log_softmax = torch.nn.LogSoftmax(dim=2)

  def init_hidden(self, batch_size):
    hidden_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    state_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    return (hidden_cell, state_cell)

  def get_embedding(self, x):
    x_embed = self.embed(x)
    
    # Total length for pad_packed_sequence method = maximum sequence length
    maximum_sequence_length = x_embed.size(1)

    return x_embed, maximum_sequence_length
  

  def forward(self, x,sentences_length,states):
    
    """
      x : bsz * seq_len
    
      hidden_encoder: ( num_lstm_layers * bsz * hidden_size, num_lstm_layers * bsz * hidden_size)

    """
    # Get Embeddings
    x_embed, maximum_padding_length = self.get_embedding(x)

    # Packing the input
    packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(input= x_embed, lengths= sentences_length, batch_first=True, enforce_sorted=False)


    packed_x_embed, states = self.lstm(packed_x_embed, states)

    x,  sentences_length = torch.nn.utils.rnn.pad_packed_sequence(packed_x_embed, batch_first=True, total_length=maximum_padding_length) # maximum_padding_length: to explicitly enforce the pad_packed_sequence layer to pad the sentences with the tallest sequence length.
  
    x = self.output(x)

    x = self.log_softmax(x)
    


    return (x, states)

  


  def inference(self, n_samples, sos):

    # generate random z 
    batch_size = 1
    length = [1,]
    idx_sample = []


    input = torch.Tensor(1, 1).fill_(self.dictionary.get_w2i()[sos]).long().to(self.device)

    hidden = self.model.init_hidden(batch_size)

    with torch.no_grad():
    
      for i in range(n_samples):
        
        pred, hidden = self.forward(input, length, hidden)
        pred = pred.exp()
        _, s = torch.topk(pred, 1)
        idx_sample.append(s.item())
        input = s.squeeze(0)

      w_sample = [self.dictionary.get_i2w()[str(idx)] for idx in idx_sample]
      w_sample = " ".join(w_sample)

    return w_sample
