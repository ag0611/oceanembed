import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'acoustic'))
import torch
import numpy as np

print("=== TEST SUITE: OceanEmbed ===\n")
passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name} -- {e}")
        failed += 1

from svp import compute_svp_profile, mackenzie_svp
from sld import compute_sld
from sofar import compute_sofar_axis
from shadow_zone import compute_shadow_zone

depths = [0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000]
temps  = [28,27,26,25,24,22,20,18,16,14,12,10,8,6,4]
sals   = [35]*15

def test_svp():
    svp = compute_svp_profile(temps, sals, depths)
    assert len(svp) == 15
    assert all(1400 < v < 1600 for v in svp)

def test_sld():
    svp = compute_svp_profile(temps, sals, depths)
    d, v = compute_sld(svp, depths)
    assert d == 0
    assert v > 1500

def test_sofar():
    svp = compute_svp_profile(temps, sals, depths)
    d, v = compute_sofar_axis(svp, depths)
    assert d == 1000
    assert v < 1500

def test_shadow_zone():
    svp = compute_svp_profile(temps, sals, depths)
    sld_d, _ = compute_sld(svp, depths)
    zones = compute_shadow_zone(svp, depths, sld_d)
    assert len(zones) == 14
    assert all('range_km' in z for z in zones)

print("[ Acoustic ]")
test("SVP profile shape & range", test_svp)
test("SLD at surface", test_sld)
test("SOFAR at deepest", test_sofar)
test("Shadow zones count", test_shadow_zone)

from encoder_cnn import CNNEncoder
from encoder_vit import ViTEncoder
from decoder import DualEncoderWithDecoder
from loss import DepthWeightedHuberLoss
from thermocline_head import ThermoclineHead

def test_cnn_encoder():
    m = CNNEncoder(in_channels=5, embedding_dim=512)
    x = torch.randn(4, 5, 64, 64)
    z = m(x)
    assert z.shape == (4, 512)

def test_vit_encoder():
    m = ViTEncoder(in_channels=5, embedding_dim=512)
    x = torch.randn(4, 5, 64, 64)
    z = m(x)
    assert z.shape == (4, 512)

def test_decoder():
    cnn = CNNEncoder(in_channels=5, embedding_dim=512)
    vit = ViTEncoder(in_channels=5, embedding_dim=512)
    m   = DualEncoderWithDecoder(cnn, vit, 15)
    x   = torch.randn(4, 5, 64, 64)
    T   = m(x)
    assert T.shape == (4, 15)

def test_loss():
    fn = DepthWeightedHuberLoss()
    p  = torch.randn(4, 15)
    t  = torch.randn(4, 15)
    l  = fn(p, t)
    assert l.item() >= 0

def test_thermocline():
    h  = ThermoclineHead(15)
    T  = torch.randn(4, 15)
    d, g = h(T)
    assert d.shape == (4,)
    assert g.shape == (4,)

print("\n[ Model ]")
test("CNN encoder output shape", test_cnn_encoder)
test("ViT encoder output shape", test_vit_encoder)
test("Dual encoder + decoder output shape", test_decoder)
test("Loss non-negative", test_loss)
test("Thermocline head shapes", test_thermocline)

print(f"\n{'='*35}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*35}")