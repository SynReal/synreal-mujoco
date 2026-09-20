# Spring panel

Open `main.json` using **File > Open** in the project panel. The scene loads
with one MJCF entity referencing `assets/spring_panel.xml`. The project UI
starts paused by default; uncheck **Pause** to run the simulation.

The orange panel is a 1 kg box measuring 0.50 × 0.36 × 0.08 m. Its center starts
0.8 m above the floor, and its free joint permits all three translations and
all three rotations. Gravity makes it fall into a fixed blue U-shaped holder,
with its center settling approximately 0.10 m above the ground. The holder
has a bottom and two side walls built from separate box geometries in one
body without joints. Its channel is 0.51 m wide, leaving 5 mm clearance on
each side of the panel. The bottom surface is at z = 0.06 m and the wall tops
are at z = 0.42 m. The channel is open at the top and both ends along Y;
contact and friction seat the panel without locking its free joint.

MuJoCo simulates the box and its contacts with the holder;
the existing scene stepper synchronizes its pose to Style3D. The floor is an
MJCF plane at z = 0, defined directly under `worldbody`; its collision surface
is infinite, while its size controls the visible extent. The current Style3D
rigid-body importer skips planes, so floor contact is handled by MuJoCo.

The project name is `spring_panel`; no spring constraint is applied. Adjust
the box position, dimensions, mass, or friction in `assets/spring_panel.xml`,
then reopen `main.json` to restart the simulation.
