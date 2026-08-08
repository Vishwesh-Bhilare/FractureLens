from fracturelens.core.io import decode_label

def test_decode_label_boundaries():
    assert [decode_label(x) for x in [1,10,11,20,21,30]] == [(1,1),(1,10),(2,1),(2,10),(3,1),(3,10)]
