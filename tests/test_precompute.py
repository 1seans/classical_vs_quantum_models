from precompute import build_data


def test_build_and_load_bs_surface(tmp_path):
    build_data.build_bs_surface(out_dir=str(tmp_path))
    data = build_data.load("bs_surface", out_dir=str(tmp_path))
    assert data["Z"].shape[0] == len(data["T_range"])
    assert data["Z"].shape[1] == len(data["vol_range"])
