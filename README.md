# p-Blend
Code and data for "p-Blend: Privacy- and Utility-Preserving Blendshape Perturbation Against Re-identification Attacks in Virtual Reality"

Dataset Access
The dataset used in this research can be downloaded from the following link: https://1drv.ms/u/c/ab8db1e1fd9a0c0c/EdJpnH7g3hdJn9pK2xdXCSABPt8vsCXbsohKEYaprE4r4w?e=Qk6fVA

Anonymization and IRB Approval
The data has been anonymized to protect privacy. The experiments involving human participants were approved by the IRB council of Shandong University and were conducted with the consent of the participants.


# p-Blend Dataset: Data Structure Overview

## Dataset Description

The **p-Blend Dataset** consists of **52-dimensional facial blendshape data** collected from users interacting with various VR applications. The dataset is organized by **user**, **session**, and **application**, with each entry representing the facial expression at a specific **time step** during the VR experience.

## Data File Format

Each file is structured as follows:
1. **Time Step Index**: An integer representing the time step.
2. **Blendshape Data**: A comma-separated list of 52 floating-point values representing the facial blendshape at that time step.
3. **Empty Line**: Separates each time step’s data.

### Example Format:

1
0.077173, 0.468907, ..., 0.077173
(empty line)

2
0.111651, 0.469502, ..., 0.048681
(empty line)

## Directory Structure

The dataset is organized by user, session, and application:

p_blend_dataset/
└── 1/ # User 1 data
├── session_one/ # First session data
│ ├── Archery/ # Archery app data
│ │ └── blendshape_data.txt
│ └── 360pano/
├── session_two/
└── ...
