# Spring panel

Open `main.json` using **File > Open** in the project panel. The scene loads
with one MJCF entity referencing `assets/spring_panel.xml`. The project UI
starts paused by default; uncheck **Pause** to run the simulation.

The orange panel is a 1 kg box measuring 10 × 10 × 0.2 cm. Its center starts
8 cm above the floor, and its free joint permits all three translations and
all three rotations. Gravity makes it fall into a fixed blue four-wall holder,
with its center settling approximately 0.70 cm above the ground. The holder
has a bottom and four walls built from separate box geometries in one
body without joints. Its cavity measures 10.2 × 10.2 cm, leaving 1 mm clearance
on each side of the panel. The walls and base are 6 mm thick. The walls rise
2.5 mm above the bottom surface at z = 0.006 m, putting their tops at z = 0.0085 m.
The holder is open only at the top;
contact and friction seat the panel without locking its free joint.
The contact time constant is 5 ms to keep impact penetration small relative
to the thin panel and holder base.

MuJoCo simulates the box and its contacts with the holder;
the existing scene stepper synchronizes its pose to Style3D. The floor is an
MJCF plane at z = 0, defined directly under `worldbody`; its collision surface
is infinite, while its size controls the visible extent. The current Style3D
rigid-body importer skips planes, so floor contact is handled by MuJoCo.

The project name is `spring_panel`; no spring constraint is applied. Adjust
the box position, dimensions, mass, or friction in `assets/spring_panel.xml`,
then reopen `main.json` to restart the simulation.
