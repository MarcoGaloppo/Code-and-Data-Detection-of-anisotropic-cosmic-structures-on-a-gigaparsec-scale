#!/usr/bin/env python3

import numpy as np
import glob
import os
import struct
import random

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

Lbox = 2000.0
DZ   = 40.0
R    = 200.0
N_target = 27685

N_circle_per_slice = 20   # number of mocks extracted per slice

snapshot_files = sorted(glob.glob("snapshot_0100.*"))

output_dir = "LCDM_mock_samples"

os.makedirs(output_dir, exist_ok=True)

if len(snapshot_files) == 0:
    raise RuntimeError("No snapshot files found!")

print("Found",len(snapshot_files),"snapshot files")

# --------------------------------------------------
# NUMBER OF Z SLICES
# --------------------------------------------------

Ns = int(Lbox / DZ)

print("Number of slices =",Ns)

mock_id = 0


# --------------------------------------------------
# LOOP OVER SLICES
# --------------------------------------------------

for s in range(Ns):

    zmin = -Lbox/2 + s*DZ
    zmax = zmin + DZ

    print("\nProcessing slice",s,"z =",zmin,"-",zmax)

    slice_particles = []


    # --------------------------------------------------
    # STREAM READ SNAPSHOT FILES
    # --------------------------------------------------

    for fname in snapshot_files:

        with open(fname,"rb") as f:

            f.read(4)
            header = f.read(256)
            f.read(4)

            npart = struct.unpack("6I", header[0:24])
            n_dm = npart[1]

            f.read(4)
            coords = np.fromfile(f, dtype=np.float32, count=3*sum(npart))
            f.read(4)

            coords = coords.reshape((-1,3))

            start = npart[0]
            end   = start + n_dm

            dm = coords[start:end]


            # --------------------------------------------------
            # CENTER THE BOX
            # --------------------------------------------------

            dm[:,0] -= Lbox/2
            dm[:,1] -= Lbox/2
            dm[:,2] -= Lbox/2


            # --------------------------------------------------
            # SELECT Z SLICE
            # --------------------------------------------------

            mask = (dm[:,2] >= zmin) & (dm[:,2] < zmax)

            if np.any(mask):

                slice_particles.append(dm[mask])


    if len(slice_particles) == 0:
        continue

    slice_particles = np.vstack(slice_particles)

    print("Particles in slice:",len(slice_particles))


    # --------------------------------------------------
    # GENERATE MULTIPLE MOCKS IN THIS SLICE
    # --------------------------------------------------

    for k in range(N_circle_per_slice):

        xc = random.uniform(-Lbox/2 + R, Lbox/2 - R)
        yc = random.uniform(-Lbox/2 + R, Lbox/2 - R)

        dx = slice_particles[:,0] - xc
        dy = slice_particles[:,1] - yc

        r = np.sqrt(dx*dx + dy*dy)

        circle_mask = r < R

        circle = slice_particles[circle_mask]

        if len(circle) < N_target:
            continue


        # ----------------------------------------------
        # RANDOM SUBSAMPLING
        # ----------------------------------------------

        idx = np.random.choice(len(circle), N_target, replace=False)

        sample = circle[idx].copy()


        # ----------------------------------------------
        # RECENTER THE SAMPLE
        # ----------------------------------------------

        sample[:,0] -= xc
        sample[:,1] -= yc


        # ----------------------------------------------
        # WRITE FILE
        # ----------------------------------------------

        outfile = os.path.join(
            output_dir,
            f"mock_{mock_id:04d}.dat"
        )

        np.savetxt(
            outfile,
            sample,
            fmt="%.6e",
            header="x y z"
        )

        print("Written",outfile)

        mock_id += 1


print("\nDone.")
print("Total mocks generated:",mock_id)
