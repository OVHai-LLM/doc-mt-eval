import torch
import torch.nn as nn

class LayerWiseAttention(nn.Module):
	"""Parameterised scalar mixture of N layer tensors, following Unbabel COMET:
	mixture = gamma * sum_i softmax(alpha)_i * tensor_i
	https://github.com/Unbabel/COMET/blob/master/comet/modules/layerwise_attention.py

	Token-level output [B, T, C]; sentence pooling is done in the model's forward.
	"""

	def __init__(self, N, layer_dropout_p=0.1, layer_norm=True, layer_transformation="sparsemax"):
		super().__init__()
		self.N = N
		self.layer_dropout_p = layer_dropout_p
		self.layer_norm = layer_norm

        # init learnable params
		self.alpha = nn.Parameter(torch.zeros(self.N, dtype=torch.bfloat16))  # 0 so at first uniform weights
		self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))  # global scale, init 1.0
		
        
		if layer_transformation == "sparsemax":
			from entmax import sparsemax
			self.transform_fn = sparsemax
		elif layer_transformation == "softmax":
			self.transform_fn = torch.softmax
		else:
			raise ValueError(f"unknown layer_transformation: {layer_transformation!r}")
        
		if self.layer_dropout_p > 0:
			# Buffers (not parameters): live on the right device automatically,
			# excluded from the optimizer. persistent=False keeps them out of
			# checkpoints — they hold no learned state.
			# copied from Unbabel
			self.register_buffer("dropout_mask", torch.zeros(self.N), persistent=False)
			self.register_buffer("dropout_fill", torch.empty(self.N).fill_(-1e20), persistent=False)

	@staticmethod
	def _layer_norm(tensor, broadcast_mask, mask):
		"""
		From: https://github.com/Unbabel/COMET/blob/master/comet/modules/layerwise_attention.py (L95-L110)

		Masked layer normalisation: computes mean and variance only over
		non-padding tokens, then z-scores the full tensor.

		Args:
			tensor:         [B, T, C]  — one layer's hidden states
			broadcast_mask: [B, T, 1]  — float padding mask, broadcast-ready
			mask:           [B, T]     — float padding mask

		Returns:
			normalized_tensor: [B, T, C]
		"""
        # tensor is the hidden state of one layer
		orig_dtype = tensor.dtype
		tensor = tensor.float()
		broadcast_mask = broadcast_mask.float()
		mask = mask.float()
    
		tensor_masked = tensor * broadcast_mask            # zero out padding positions
		batch_size, _, input_dim = tensor.size()
        
		num_elements_not_masked = mask.sum(1) * input_dim  # [B]

		# Mean over real elements only
		mean = tensor_masked.view(batch_size, -1).sum(1)   # [B]
		mean = (mean / num_elements_not_masked).view(batch_size, 1, 1)  # [B, 1, 1]

		# Variance over real elements only
		variance = (((tensor_masked - mean) * broadcast_mask) ** 2).view(
			batch_size, -1
		).sum(1) / num_elements_not_masked                 # [B]

		# Z-score normalisation (epsilon for numerical stability)
		normalized_tensor = (tensor - mean) / torch.sqrt(variance + 1e-12).view(
			batch_size, 1, 1
		)
		return normalized_tensor.to(orig_dtype)

	def forward(self, hidden_states, attention_mask):
		# hidden_states: tuple/list of N tensors, each [B, T, C]
		assert len(hidden_states) == self.N, f"expected {self.N} layers, got {len(hidden_states)}"

		# Layer dropout (Unbabel COMET, layerwise_attention.py L121-L124)
		weights = self.alpha

		if self.training and self.layer_dropout_p > 0:
			dropout_mask = self.dropout_mask.to(weights.device)
			dropout_fill = self.dropout_fill.to(weights.device)
			weights = torch.where(
				dropout_mask.uniform_() > self.layer_dropout_p,
				weights,
				dropout_fill,
            )

		weights = self.transform_fn(weights, dim=0)  # [N] softmax or sparsemax

		######### Layer-wise mixing #########
        # As in Unbabel
		if not self.layer_norm:
			pieces = []
			for i in range(self.N):
				pieces.append(weights[i] * hidden_states[i]) # Multiplied by the weight
			out = sum(pieces)                               # [B, T, C]
		else:
			# Unbabel's layer_norm branch (layerwise_attention.py L135-L142).
			mask_float = attention_mask.float()             # [B, T] before it was just integers: 0 and 1
			broadcast_mask = mask_float.unsqueeze(-1)       # [B, T, 1] insert a new axis
			pieces = []
			for i in range(self.N):
				normed = self._layer_norm(hidden_states[i], broadcast_mask, mask_float)  # [B, T, C]
				pieces.append(weights[i] * normed)
			out = sum(pieces)                                  # [B, T, C]

		out = self.gamma * out                              # global scale
		return out
