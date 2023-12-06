class TransformerDecoderLayer(Module):
    # ... (rest of the code remains unchanged)

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        # ... (rest of the code remains unchanged)
        self._sa_weights = None
        self._mha_weights = None

    # ... (rest of the code remains unchanged)

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
        # ... (rest of the code remains unchanged)

        if self.norm_first:
            sa_output, self._sa_weights = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            mha_output, self._mha_weights = self._mha_block(self.norm2(sa_output), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            ff_output = self._ff_block(self.norm3(mha_output))
        else:
            sa_output, self._sa_weights = self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            mha_output, self._mha_weights = self._mha_block(sa_output, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            ff_output = self._ff_block(mha_output)

        return ff_output, self._sa_weights, self._mha_weights

    # ... (rest of the code remains unchanged)

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        self_attn_output, self._sa_weights = self.self_attn(x, x, x,
                                                            attn_mask=attn_mask,
                                                            key_padding_mask=key_padding_mask,
                                                            is_causal=is_causal,
                                                            need_weights=True)
        return self_attn_output, self._sa_weights

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        multihead_attn_output, self._mha_weights = self.multihead_attn(x, mem, mem,
                                                                       attn_mask=attn_mask,
                                                                       key_padding_mask=key_padding_mask,
                                                                       is_causal=is_causal,
                                                                       need_weights=True)
        return multihead_attn_output, self._mha_weights
