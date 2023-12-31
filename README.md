# tstream
 Using transformers to model stream flow from meteorological data.

## PyTorch changes to extract attention weights
 In order to be able to extract the attention weights from the decoder I have done some changes to the <code>torch.nn</code> module.

 1. Added <code>AttentionWeightsTransformerDecoderLayer</code> class to the <code>torch.nn.modules.transformer.py</code> file. This class is based on the original <code>TransformerDecoderLayer</code> but it has been customized to return the attention weights. Here is the code added.
    
        class AttentionWeightsTransformerDecoderLayer(Module):
            r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
            This standard decoder layer is based on the paper "Attention Is All You Need".
            Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
            Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
            Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
            in a different way during application.

            Args:
                d_model: the number of expected features in the input (required).
                nhead: the number of heads in the multiheadattention models (required).
                dim_feedforward: the dimension of the feedforward network model (default=2048).
                dropout: the dropout value (default=0.1).
                activation: the activation function of the intermediate layer, can be a string
                    ("relu" or "gelu") or a unary callable. Default: relu
                layer_norm_eps: the eps value in layer normalization components (default=1e-5).
                batch_first: If ``True``, then the input and output tensors are provided
                    as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
                norm_first: if ``True``, layer norm is done prior to self attention, multihead
                    attention and feedforward operations, respectively. Otherwise it's done after.
                    Default: ``False`` (after).
                bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
                    bias. Default: ``True``.

            Examples::
                >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
                >>> memory = torch.rand(10, 32, 512)
                >>> tgt = torch.rand(20, 32, 512)
                >>> out = decoder_layer(tgt, memory)

            Alternatively, when ``batch_first`` is ``True``:
                >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
                >>> memory = torch.rand(32, 10, 512)
                >>> tgt = torch.rand(32, 20, 512)
                >>> out = decoder_layer(tgt, memory)
            """
            __constants__ = ['norm_first']

            def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                        layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                        bias: bool = True, device=None, dtype=None) -> None:
                factory_kwargs = {'device': device, 'dtype': dtype}
                super().__init__()
                self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                    bias=bias, **factory_kwargs)
                self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                        bias=bias, **factory_kwargs)
                
                # Define objects to store custom weights
                self._sa_weights = None
                self._mha_weights = None
                
                # Implementation of Feedforward model
                self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
                self.dropout = Dropout(dropout)
                self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

                self.norm_first = norm_first
                self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
                self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
                self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
                self.dropout1 = Dropout(dropout)
                self.dropout2 = Dropout(dropout)
                self.dropout3 = Dropout(dropout)

                # Legacy string support for activation function.
                if isinstance(activation, str):
                    self.activation = _get_activation_fn(activation)
                else:
                    self.activation = activation

            def __setstate__(self, state):
                if 'activation' not in state:
                    state['activation'] = F.relu
                super().__setstate__(state)

            def forward(
                self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False,
            ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
                r"""Pass the inputs (and mask) through the decoder layer.

                Args:
                    tgt: the sequence to the decoder layer (required).
                    memory: the sequence from the last layer of the encoder (required).
                    tgt_mask: the mask for the tgt sequence (optional).
                    memory_mask: the mask for the memory sequence (optional).
                    tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                    memory_key_padding_mask: the mask for the memory keys per batch (optional).
                    tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                        Default: ``False``.
                        Warning:
                        ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                        the causal mask. Providing incorrect hints can result in
                        incorrect execution, including forward and backward
                        compatibility.
                    memory_is_causal: If specified, applies a causal mask as
                        ``memory mask``.
                        Default: ``False``.
                        Warning:
                        ``memory_is_causal`` provides a hint that
                        ``memory_mask`` is the causal mask. Providing incorrect
                        hints can result in incorrect execution, including
                        forward and backward compatibility.

                Shape:
                    see the docs in Transformer class.
                """
                # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

                x = tgt
                if self.norm_first:
                    x = self.norm1(x)
                    tmp_x_sa, self._sa_weights = self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
                    x = x + tmp_x_sa
                    x = self.norm2(x)
                    temp_x_mha, self._mha_weights = self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
                    x = x + temp_x_mha
                    x = x + self._ff_block(self.norm3(x))
                else:
                    tmp_x_sa, self._sa_weights = self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
                    x = self.norm1(x + tmp_x_sa)
                    temp_x_mha, self._mha_weights = self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
                    x = self.norm2(x + temp_x_mha)
                    x = self.norm3(x + self._ff_block(x))

                return x, self._sa_weights, self._mha_weights

            # self-attention block
            def _sa_block(self, x: Tensor,
                        attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
                x, self._sa_weights = self.self_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=True)
                return self.dropout1(x), self._sa_weights

            # multihead attention block
            def _mha_block(self, x: Tensor, mem: Tensor,
                        attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
                x, self._mha_weights = self.multihead_attn(x, mem, mem,
                                        attn_mask=attn_mask,
                                        key_padding_mask=key_padding_mask,
                                        is_causal=is_causal,
                                        need_weights=True)
                return self.dropout2(x), self._mha_weights

            # feed forward block
            def _ff_block(self, x: Tensor) -> Tensor:
                x = self.linear2(self.dropout(self.activation(self.linear1(x))))
                return self.dropout3(x)


        def _get_clones(module, N):
            # FIXME: copy.deepcopy() is not defined on nn.module
            return ModuleList([copy.deepcopy(module) for i in range(N)])


        def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
            if activation == "relu":
                return F.relu
            elif activation == "gelu":
                return F.gelu

            raise RuntimeError(f"activation should be relu/gelu, not {activation}")


        def _detect_is_causal_mask(
                mask: Optional[Tensor],
                is_causal: Optional[bool] = None,
                size: Optional[int] = None,
        ) -> bool:
            """Return whether the given attention mask is causal.

            Warning:
            If ``is_causal`` is not ``None``, its value will be returned as is.  If a
            user supplies an incorrect ``is_causal`` hint,

            ``is_causal=False`` when the mask is in fact a causal attention.mask
            may lead to reduced performance relative to what would be achievable
            with ``is_causal=True``;
            ``is_causal=True`` when the mask is in fact not a causal attention.mask
            may lead to incorrect and unpredictable execution - in some scenarios,
            a causal mask may be applied based on the hint, in other execution
            scenarios the specified mask may be used.  The choice may not appear
            to be deterministic, in that a number of factors like alignment,
            hardware SKU, etc influence the decision whether to use a mask or
            rely on the hint.
            ``size`` if not None, check whether the mask is a causal mask of the provided size
            Otherwise, checks for any causal mask.
            """
            # Prevent type refinement
            make_causal = (is_causal is True)

            if is_causal is None and mask is not None:
                sz = size if size is not None else mask.size(-2)
                causal_comparison = _generate_square_subsequent_mask(
                    sz, device=mask.device, dtype=mask.dtype)

                # Do not use `torch.equal` so we handle batched masks by
                # broadcasting the comparison.
                if mask.size() == causal_comparison.size():
                    make_causal = bool((mask == causal_comparison).all())
                else:
                    make_causal = False

            return make_causal
 
 2. The <code>__init__</code> method of the <code>TransformerDecoder</code> class has been changed to add the new <code>AttentionWeightsTransformerDecoderLayer</code> class as the last layer. If this was not changed and the user would like to use several decoder layers, the first layer would pass the attention matrix and also the sa_weights and mhe_weights to the following layer causing a error because it is only supposed to pass the attention matrix. This way all the layers are conventional decoder layers but the last one which uses the new class to return the weights.
        
            def __init__(self, decoder_layer, num_layers, decoder_layer_attention_weights, num_layers_attention_weights, norm=None):
                super().__init__()
                torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
                self.layers = _get_clones(decoder_layer, num_layers) + _get_clones(decoder_layer_attention_weights, num_layers_attention_weights)
                self.num_layers = num_layers
                self.num_layers_attention_weights = num_layers_attention_weights
                self.norm = norm

 3. The type hint type <code>Tuple</code> has to be added in the second line of the <code>torch.nn.modules.transformer.py</code> file and the class name <code>AttentionWeightsTransformerDecoderLayer</code> has to be included in the <code>__all__</code> statement.
        
        from typing import Optional, Any, Union, Callable, Tuple

        __all__ = ['Transformer', 'TransformerEncoder', 'TransformerDecoder', 'TransformerEncoderLayer', 'TransformerDecoderLayer', 'AttentionWeightsTransformerDecoderLayer']

 4. the <code>torch.nn.modules.__init__.py</code> file was modified accordingly. In particular, the class name <code>AttentionWeightsTransformerDecoderLayer</code> has to be added to the <code>__all__</code> statement.
    
            __all__ = [
                'Module', 'Identity', 'Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d',
                'ConvTranspose2d', 'ConvTranspose3d', 'Threshold', 'ReLU', 'Hardtanh', 'ReLU6',
                'Sigmoid', 'Tanh', 'Softmax', 'Softmax2d', 'LogSoftmax', 'ELU', 'SELU', 'CELU', 'GLU', 'GELU', 'Hardshrink',
                'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention', 'PReLU', 'Softsign', 'Softmin',
                'Tanhshrink', 'RReLU', 'L1Loss', 'NLLLoss', 'KLDivLoss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss',
                'NLLLoss2d', 'PoissonNLLLoss', 'CosineEmbeddingLoss', 'CTCLoss', 'HingeEmbeddingLoss', 'MarginRankingLoss',
                'MultiLabelMarginLoss', 'MultiLabelSoftMarginLoss', 'MultiMarginLoss', 'SmoothL1Loss', 'GaussianNLLLoss',
                'HuberLoss', 'SoftMarginLoss', 'CrossEntropyLoss', 'Container', 'Sequential', 'ModuleList', 'ModuleDict',
                'ParameterList', 'ParameterDict', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'MaxPool1d', 'MaxPool2d',
                'MaxPool3d', 'MaxUnpool1d', 'MaxUnpool2d', 'MaxUnpool3d', 'FractionalMaxPool2d', "FractionalMaxPool3d",
                'LPPool1d', 'LPPool2d', 'LocalResponseNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'InstanceNorm1d',
                'InstanceNorm2d', 'InstanceNorm3d', 'LayerNorm', 'GroupNorm', 'SyncBatchNorm',
                'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout',
                'ReflectionPad1d', 'ReflectionPad2d', 'ReflectionPad3d', 'ReplicationPad2d', 'ReplicationPad1d', 'ReplicationPad3d',
                'CrossMapLRN2d', 'Embedding', 'EmbeddingBag', 'RNNBase', 'RNN', 'LSTM', 'GRU', 'RNNCellBase', 'RNNCell',
                'LSTMCell', 'GRUCell', 'PixelShuffle', 'PixelUnshuffle', 'Upsample', 'UpsamplingNearest2d', 'UpsamplingBilinear2d',
                'PairwiseDistance', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d', 'AdaptiveAvgPool1d',
                'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d', 'TripletMarginLoss', 'ZeroPad1d', 'ZeroPad2d', 'ZeroPad3d',
                'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d', 'Bilinear', 'CosineSimilarity', 'Unfold', 'Fold',
                'AdaptiveLogSoftmaxWithLoss', 'TransformerEncoder', 'TransformerDecoder',
                'TransformerEncoderLayer', 'TransformerDecoderLayer', 'AttentionWeightsTransformerDecoderLayer', 'Transformer',
                'LazyLinear', 'LazyConv1d', 'LazyConv2d', 'LazyConv3d',
                'LazyConvTranspose1d', 'LazyConvTranspose2d', 'LazyConvTranspose3d',
                'LazyBatchNorm1d', 'LazyBatchNorm2d', 'LazyBatchNorm3d',
                'LazyInstanceNorm1d', 'LazyInstanceNorm2d', 'LazyInstanceNorm3d',
                'Flatten', 'Unflatten', 'Hardsigmoid', 'Hardswish', 'SiLU', 'Mish', 'TripletMarginWithDistanceLoss', 'ChannelShuffle',
                'CircularPad1d', 'CircularPad2d', 'CircularPad3d'
            ]
