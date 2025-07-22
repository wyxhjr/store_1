from typing import Optional
import torch
from torch import Tensor
from DistEmb import DistEmbedding

def myEmb_init_data(shape, dtype):
    return torch.nn.init.normal_(torch.empty(size=shape, dtype=dtype, device='cpu'))
class myEmbeddingBag:
    r"""Computes sums or means of 'bags' of embeddings, without instantiating the
    intermediate embeddings.

    For bags of constant length, no :attr:`per_sample_weights`, no indices equal to :attr:`padding_idx`,
    and with 2D inputs, this class

        * with ``mode="sum"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.sum(dim=1)``,
        * with ``mode="mean"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.mean(dim=1)``,
        * with ``mode="max"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.max(dim=1)``.

    However, :class:`~torch.nn.EmbeddingBag` is much more time and memory efficient than using a chain of these
    operations.

    EmbeddingBag also supports per-sample weights as an argument to the forward
    pass. This scales the output of the Embedding before performing a weighted
    reduction as specified by ``mode``. If :attr:`per_sample_weights` is passed, the
    only supported ``mode`` is ``"sum"``, which computes a weighted sum according to
    :attr:`per_sample_weights`.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (str, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 ``"sum"`` computes the weighted sum, taking :attr:`per_sample_weights`
                                 into consideration. ``"mean"`` computes the average of the values
                                 in the bag, ``"max"`` computes the max value over each bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. See
                                 Notes for more details regarding sparse gradients. Note: this option is not
                                 supported when ``mode="max"``.
        include_last_offset (bool, optional): if ``True``, :attr:`offsets` has one additional element, where the last element
                                      is equivalent to the size of `indices`. This matches the CSR format.
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the
                                     gradient; therefore, the embedding vector at :attr:`padding_idx` is not updated
                                     during training, i.e. it remains as a fixed "pad". For a newly constructed
                                     EmbeddingBag, the embedding vector at :attr:`padding_idx` will default to all
                                     zeros, but can be updated to another value to be used as the padding vector.
                                     Note that the embedding vector at :attr:`padding_idx` is excluded from the
                                     reduction.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape `(num_embeddings, embedding_dim)`
                         initialized from :math:`\mathcal{N}(0, 1)`.

    Examples::

        >>> # an EmbeddingBag module containing 10 tensors of size 3
        >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([1,2,4,5,4,3,2,9], dtype=torch.long)
        >>> offsets = torch.tensor([0,4], dtype=torch.long)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> embedding_sum(input, offsets)
        tensor([[-0.8861, -5.4350, -0.0523],
                [ 1.1306, -2.5798, -1.0044]])

        >>> # Example with padding_idx
        >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum', padding_idx=2)
        >>> input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9], dtype=torch.long)
        >>> offsets = torch.tensor([0,4], dtype=torch.long)
        >>> embedding_sum(input, offsets)
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7082,  3.2145, -2.6251]])

        >>> # An EmbeddingBag can be loaded from an Embedding like so
        >>> embedding = nn.Embedding(10, 3, padding_idx=2)
        >>> embedding_sum = nn.EmbeddingBag.from_pretrained(
                embedding.weight,
                padding_idx=embedding.padding_idx,
                mode='sum')
    """
    __constants__ = ['num_embeddings', 'embedding_dim', 'max_norm', 'norm_type',
                     'scale_grad_by_freq', 'mode', 'sparse', 'include_last_offset',
                     'padding_idx']

    num_embeddings: int
    embedding_dim: int
    weight: DistEmbedding
    mode: str
    sparse: bool
    include_last_offset: bool
    padding_idx: Optional[int]

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 mode: str = 'mean', sparse: bool = False, _weight: Optional[Tensor] = None,
                 include_last_offset: bool = False, padding_idx: Optional[int] = None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        if _weight is None:
            self.weight = DistEmbedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim,init_func=myEmb_init_data)
            #Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
            self.reset_parameters()
        else:
            assert(0), 'TODO'
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            #self.weight = Parameter(_weight)
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset

    def reset_parameters(self) -> None:
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def calu(self, emb):
        if self.mode == 'sum':
            ret = emb.sum(dim=0)
        elif self.mode == 'mean':
            ret = emb.mean(dim=0)
        else:
            assert self.mode == 'max', f'not supported mode \'{self.mode}\''
            ret = emb.max(dim=0)
        return ret

    def __call__(self, input: Tensor, offsets: Optional[Tensor] = None, per_sample_weights: Optional[Tensor] = None) -> Tensor:
        """Forward pass of EmbeddingBag.

        Args:
            input (Tensor): Tensor containing bags of indices into the embedding matrix.
            offsets (Tensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                the starting index position of each bag (sequence) in :attr:`input`.
            per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
                to indicate all weights should be taken to be ``1``. If specified, :attr:`per_sample_weights`
                must have exactly the same shape as input and is treated as having the same
                :attr:`offsets`, if those are not ``None``. Only supported for ``mode='sum'``.

        Returns:
            Tensor output shape of `(B, embedding_dim)`.

        .. note::

            A few notes about ``input`` and ``offsets``:

            - :attr:`input` and :attr:`offsets` have to be of the same type, either int or long

            - If :attr:`input` is 2D of shape `(B, N)`, it will be treated as ``B`` bags (sequences)
              each of fixed length ``N``, and this will return ``B`` values aggregated in a way
              depending on the :attr:`mode`. :attr:`offsets` is ignored and required to be ``None`` in this case.

            - If :attr:`input` is 1D of shape `(N)`, it will be treated as a concatenation of
              multiple bags (sequences).  :attr:`offsets` is required to be a 1D tensor containing the
              starting index positions of each bag in :attr:`input`. Therefore, for :attr:`offsets` of shape `(B)`,
              :attr:`input` will be viewed as having ``B`` bags. Empty bags (i.e., having 0-length) will have
              returned vectors filled by zeros.
        """
        emb = self.weight(input)
        if per_sample_weights != None:
            emb = emb * per_sample_weights.view(-1,1)
        if  offsets != None:
            #assert len(offsets)
            ret = torch.empty((len(offsets),self.embedding_dim))
            for i in range(len(offsets)):
                st = offsets[i]
                ed = offsets[i+1] if i + 1 < len(offsets) else self.num_embeddings
                ret[i] = self.calu(emb[st:ed])
        else:
            ret = self.calu(emb)
        return ret