import laspy
import numpy as np

las = laspy.read("/home/sabbir/My_Projects/SegFormer/inference/input/2026-03-31_262218_1Cloud.laz")

# Find the verticality field
all_dims = [d.name for d in las.point_format.dimensions]
#extra_dims = [d.name for d in las.point_format.extra_dims]
print("All dims:", all_dims )

# Read the verticality values
vert = np.array(las["Verticality (0.05)"], dtype=np.float32)
print(f"\nVerticality field stats:")
print(f"  dtype  : {las['Verticality (0.05)'].dtype}")
print(f"  min    : {vert.min():.6f}")
print(f"  max    : {vert.max():.6f}")
print(f"  mean   : {vert.mean():.6f}")
print(f"  zeros  : {(vert == 0).sum():,}  ({100*(vert==0).mean():.1f}%)")
print(f"  >1.0   : {(vert > 1.0).sum():,}")
print(f"  sample : {vert[:10]}")