"""Implements common unimodal encoders."""
import torch
import math
from torch import nn
import torch.nn.functional as F

class Linear(torch.nn.Module):
    """Linear Layer with Xavier Initialization, and 0 Bias."""
    
    def __init__(self, indim, outdim, xavier_init=False):
        """Initialize Linear Layer w/ Xavier Init.

        Args:
            indim (int): Input Dimension
            outdim (int): Output Dimension
            xavier_init (bool, optional): Whether to apply Xavier Initialization to Layer. Defaults to False.
        
        """
        super(Linear, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        if xavier_init:
            nn.init.xavier_normal(self.fc.weight)
            self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        """Apply Linear Layer to Input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor
        
        """
        return self.fc(x)



class Transformer(nn.Module):
    """Extends nn.Transformer."""
    
    def __init__(self, n_features, dim, nhead=5, num_layers=5, conv1d=True, out_last=True, pos_embd=False, pos_learnable=False, max_len=128):
        """Initialize Transformer object.

        Args:
            n_features (int): Number of features in the input.
            dim (int): Dimension which to embed upon / Hidden dimension size.
        """
        super().__init__()
        self.embed_dim = dim
        self.conv1d = conv1d
        self.out_last = out_last
        self.pos_embd = pos_embd
        self.pos_learnable = pos_learnable
        self.max_len = max_len
        if self.conv1d:
            self.conv = nn.Conv1d(n_features, self.embed_dim,
                                kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Positional embeddings (optional)
        if self.pos_embd:
            if self.pos_learnable:
                self.pos_embedding = nn.Embedding(self.max_len, self.embed_dim)
            else:
                # fixed sinusoidal table
                position = torch.arange(self.max_len).unsqueeze(1)  # (L,1)
                div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim))
                pe = torch.zeros(self.max_len, self.embed_dim)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pos_table', pe)  # (L, D)

    def forward(self, x, lengths=None):
        """Apply Transformer to Input.

        Args:
            x (torch.Tensor): Layer Input
            lengths (torch.Tensor, optional): (batch_size) - Optional, the true lengths of each sequence in the batch.

        Returns:
            torch.Tensor: Layer Output
        """
        if type(x) is list:
            x = x[0] # (batch_size, seq_len, embed_dim)
        
        batch_size, seq_len, _ = x.shape
        
        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)

        if self.conv1d:
            x = self.conv(x.permute([0, 2, 1])) # (batch_size, embed_dim/channel, seq_len)
            x = x.permute([2, 0, 1]) # (seq_len, batch_size, embed_dim)
        else:
            x = x.permute([1, 0, 2]) # (seq_len, batch_size, embed_dim)
        
        # Add positional embeddings if requested
        if self.pos_embd:
            seq_len = x.size(0)
            if seq_len > self.max_len:
                seq_len = self.max_len
                x = x[:seq_len]
            curr_seq_len = x.size(0)
            positions = torch.arange(curr_seq_len, device=x.device)
            if self.pos_learnable:
                pos = self.pos_embedding(positions)  # (seq_len, D)
            else:
                pos = self.pos_table[positions]      # (seq_len, D)
            x = x + pos.unsqueeze(1)  # (seq_len, batch, D)
        
        # Create causal mask for autoregressive modeling
        seq_len = x.size(0)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask, is_causal=True)
        if self.out_last:
            if lengths is not None:
                # x: (seq_len, batch, embed_dim)
                x = x.permute(1, 0, 2) # (batch, seq_len, embed_dim)
                batch_indices = torch.arange(batch_size, device=x.device)
                last_indices = lengths - 1
                return x[batch_indices, last_indices, :]
            return x[-1]
        x = x.permute([1, 0, 2]) # (batch_size, seq_len, embed_dim)
        return x

class MSE(nn.Module):
	def __init__(self):
		super(MSE, self).__init__()
	
	def forward(self, predictions, targets, mask=None):
		"""
        Args:
            predictions: (Batch, Time, Dim) 
            targets: (Batch, Time, Dim)
            mask: (Batch, Time) - 0 for padding, 1 for valid. 
        """
		if mask is None:
			return (predictions - targets).pow(2).mean()
		mask_expanded = mask.unsqueeze(-1).expand_as(targets)
		return ((predictions - targets) ** 2 * mask_expanded.float()).sum() / (mask_expanded.float().sum()+1e-8)

class SequenceInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, mask=None):
        """
        Args:
            predictions: (Batch, Time, Dim) 
            targets: (Batch, Time, Dim)
            mask: (Batch, Time) - 0 for padding, 1 for valid. 
        """
        if mask is not None:
            # mask.bool() -> (Batch, Time)
            # predictions[mask.bool()] -> (Total_Valid_Tokens, Dim)
            valid_preds = predictions[mask.bool()]
            valid_targets = targets[mask.bool()]
        else:
            valid_preds = predictions.flatten(0, 1)
            valid_targets = targets.flatten(0, 1)

        valid_preds = F.normalize(valid_preds, dim=-1)
        valid_targets = F.normalize(valid_targets, dim=-1)

        logits = torch.matmul(valid_preds, valid_targets.T)
        logits /= self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = self.cross_entropy(logits, labels)

        return loss

# UML model
class UML(nn.Module):
	def __init__(self, xproj_in, yproj_in, shared_encoder, decoders, modality='x', infoNCE_loss=False):
		super().__init__()
		self.xproj_in = xproj_in
		self.yproj_in = yproj_in
		self.encoder = shared_encoder
		self.decoders = nn.ModuleList(decoders)
		self.modality = modality
		self.critic = MSE()
		self.infoNCE_loss = infoNCE_loss
		if self.infoNCE_loss:
			self.y_critic = SequenceInfoNCELoss()
		else:
			self.y_critic = MSE()
		print("Training using MUSE with modality: ", modality)
	
	def forward(self, x, y, x_lengths=None, y_lengths=None):
		# pool sequence dim of z
		loss_x = torch.tensor(0.0)
		loss_y = torch.tensor(0.0)
		if x is not None:
			x = x.unsqueeze(1).float() if x.ndim == 2 else x
			x_proj = self.xproj_in(x)
			zx = self.encoder(x_proj, lengths=x_lengths)
			x_recon = self.decoders[0](zx)

			mask = None
			if x_lengths is not None:
				mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < x_lengths.unsqueeze(1)
			# print((x_lengths==1).sum().item())

			if x_recon.shape[1] == 1:
				loss_x = self.critic(x_recon[:, 0, :], x[:, 0, :])
			else:
				# next embedding prediction loss
				loss_x = self.critic(x_recon[:, :-1,:], x[:,1:,:], mask=mask[:,1:] if mask is not None else None)

			diff_next_x = (x_proj - zx).pow(2).mean()

			# # k-step prediction loss
			# with torch.no_grad():
			# 	k = min(10, x_lengths.min().item() - 1 if x_lengths is not None else x.shape[1] - 1)
			# 	loss_x_k = torch.tensor(0.0)
			# 	x_recon_t = x_recon.detach()
			# 	for t in range(1,k+1):
			# 		x_proj_t = self.xproj_in(x_recon_t[:,:-1,:])
			# 		zx_t = self.encoder(x_proj_t, lengths=x_lengths-t if x_lengths is not None else None)
			# 		x_recon_t = self.decoders[0](zx_t)
			# 	loss_x_k = self.critic(x_recon_t, x[:, k:, :], mask=mask[:, k:] if mask is not None else None)
			# 	print(k)
			# 	loss_trivial_x_k = self.critic(x[:, :-k, :], x[:, k:, :], mask=mask[:, k:] if mask is not None else None)

		if y is not None:
			y = y.unsqueeze(1).float() if y.ndim == 2 else y
			y_proj = self.yproj_in(y)
			zy = self.encoder(y_proj)
			y_recon = self.decoders[1](zy)

			mask = None
			if y_lengths is not None:
				mask = torch.arange(y.shape[1], device=y.device).unsqueeze(0) < y_lengths.unsqueeze(1)

			if y_recon.shape[1] == 1:
				loss_y = self.critic(y_recon[:, 0, :], y[:, 0, :])
			else:
				loss_y = self.y_critic(y_recon[:, :-1,:], y[:,1:,:], mask=mask[:,1:] if mask is not None else None)
			diff_next_y = (y_proj - zy).pow(2).mean()

			# # k-step prediction loss
			# with torch.no_grad():
			# 	k = min(10, y.shape[1]-1)
			# 	loss_y_k = torch.tensor(0.0)
			# 	y_recon_t = y_recon.detach()
			# 	for t in range(1,k+1):
			# 		y_proj_t = self.yproj_in(y_recon_t[:, :-1, :])
			# 		zy_t = self.encoder(y_proj_t, lengths=y_lengths-t if y_lengths is not None else None)
			# 		y_recon_t = self.decoders[1](zy_t)
			# 	loss_y_k = self.critic(y_recon_t, y[:, k:, :], mask=mask[:, k:] if mask is not None else None)
			# 	loss_trivial_y_k = self.critic(y[:, :-k, :], y[:, k:, :], mask=mask[:, k:] if mask is not None else None)

		loss_private = torch.tensor(0.0, device=x.device if x is not None else y.device)
		if x is not None and y is not None:
			x_private = (x_proj - zx)
			y_private = (y_proj - zy)
			loss_private = ((x_private * y_private).mean([1,2])**2).sum()
		
		return {'loss_x': loss_x, 'loss_y': loss_y, 'loss_private': loss_private, 
			'x_proj': x_proj if x is not None else None, 'y_proj': y_proj if y is not None else None,
			'zx': zx if x is not None else None, 'zy': zy if y is not None else None, 
			'x_recon': x_recon if x is not None else None, 'y_recon': y_recon if y is not None else None,
			# 'k_step_loss_x': loss_x_k if x is not None else None, 'k_step_loss_y': loss_y_k if y is not None else None,
			# 'k_step_trivial_loss_x': loss_trivial_x_k if x is not None else None, 'k_step_trivial_loss_y': loss_trivial_y_k if y is not None else None,
			'x_private': x_private if x is not None else None, 'y_private': y_private if y is not None else None,
			'diff_next_x': diff_next_x if x is not None else None, 'diff_next_y': diff_next_y if y is not None else None}

	def get_embedding(self, x, y):
		x = x.unsqueeze(1).float() if x.ndim == 2 else x
		y = y.unsqueeze(1).float() if y.ndim == 2 else y
		x = self.xproj_in(x)
		y = self.yproj_in(y)
		return self.encoder(x).mean(dim=1), self.encoder(y).mean(dim=1)