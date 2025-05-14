from src.preference import normalize_style

def test_normalize_style():
    assert normalize_style(" Casual ") == "casual"
    assert normalize_style(" FeMiNiN ") == "feminine"
