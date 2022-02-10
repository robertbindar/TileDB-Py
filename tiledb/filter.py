from dataclasses import dataclass, field
from typing import Sequence

from tiledb import cc as lt
from .ctx import default_ctx
from .libtiledb import (
    Ctx,
    Filter,
    CompressionFilter,
    GzipFilter,
    ZstdFilter,
    LZ4Filter,
    Bzip2Filter,
    RleFilter,
    DoubleDeltaFilter,
    BitWidthReductionFilter,
    BitShuffleFilter,
    ByteShuffleFilter,
    PositiveDeltaFilter,
    ChecksumMD5Filter,
    ChecksumSHA256Filter,
    NoOpFilter,
)


@dataclass
class FilterList(lt.FilterList):
    """
    An ordered list of Filter objects for filtering TileDB data.

    FilterLists contain zero or more Filters, used for filtering attribute data, the array coordinate data, etc.

    :param ctx: A TileDB context
    :type ctx: tiledb.Ctx
    :param filters: An iterable of Filter objects to add.
    :param chunksize: (default None) chunk size used by the filter list in bytes
    :type chunksize: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     # Create several filters
    ...     gzip_filter = tiledb.GzipFilter()
    ...     bw_filter = tiledb.BitWidthReductionFilter()
    ...     # Create a filter list that will first perform bit width reduction, then gzip compression.
    ...     filters = tiledb.FilterList([bw_filter, gzip_filter])
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64, filters=filters)
    ...     # Create a second attribute filtered only by gzip compression.
    ...     a2 = tiledb.Attr(name="a2", dtype=np.int64,
    ...                      filters=tiledb.FilterList([gzip_filter]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1, a2))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    filters: Sequence[Filter] = field(default=None)
    chunksize: int = field(default=None)
    ctx: Ctx = field(default_factory=default_ctx, repr=False)

    _ctx: lt.Context = field(init=False, repr=False)

    def __post_init__(self):
        self._ctx = lt.Context(self.ctx.__capsule__(), False)

        super().__init__(self._ctx)

        if self.filters:
            for filter in self.filters:
                self.append(filter)

    # def _repr_html_(self):
    #     output = io.StringIO()

    #     output.write("<section>\n")
    #     for i in range(len(self)):
    #         output.write(self[i]._repr_html_())
    #     output.write("</section>\n")

    #     return output.getvalue()

    def append(self, filter):
        """Appends `filter` to the end of filter list

        :param filter: filter object to add
        :type filter: Filter
        :returns: None
        """

        if not isinstance(filter, Filter):
            raise ValueError("filter argument must be a TileDB filter objects")

        filtype_cls_to_enum = {
            GzipFilter: lt.FilterType.GZIP,
            ZstdFilter: lt.FilterType.ZSTD,
            LZ4Filter: lt.FilterType.LZ4,
            Bzip2Filter: lt.FilterType.BZIP2,
            RleFilter: lt.FilterType.RLE,
            DoubleDeltaFilter: lt.FilterType.DOUBLE_DELTA,
            BitWidthReductionFilter: lt.FilterType.BIT_WIDTH_REDUCTION,
            BitShuffleFilter: lt.FilterType.BITSHUFFLE,
            ByteShuffleFilter: lt.FilterType.BYTESHUFFLE,
            PositiveDeltaFilter: lt.FilterType.POSITIVE_DELTA,
            ChecksumMD5Filter: lt.FilterType.CHECKSUM_MD5,
            ChecksumSHA256Filter: lt.FilterType.CHECKSUM_SHA256,
            NoOpFilter: None,
        }
        filtype = filtype_cls_to_enum[type(filter)]
        fil = lt.Filter(self._ctx, filtype)

        if isinstance(filter, CompressionFilter):
            fil.set_option(self._ctx, lt.FilterOption.COMPRESSION_LEVEL, filter.level)
        elif isinstance(filter, BitWidthReductionFilter):
            fil.set_option(
                self._ctx, lt.FilterOption.BIT_WIDTH_MAX_WINDOW, filter.window
            )
        elif isinstance(filter, PositiveDeltaFilter):
            fil.set_option(
                self._ctx, lt.FilterOption.POSITIVE_DELTA_MAX_WINDOW, filter.window
            )

        self.add_filter(fil)
