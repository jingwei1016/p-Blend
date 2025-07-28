# p-Blend
Code and data for "p-Blend: Privacy- and Utility-Preserving Blendshape Perturbation Against Re-identification Attacks in Virtual Reality"

Dataset Access
The dataset used in this research can be downloaded from the following link: https://1drv.ms/u/c/ab8db1e1fd9a0c0c/EdJpnH7g3hdJn9pK2xdXCSABPt8vsCXbsohKEYaprE4r4w?e=Qk6fVA


# p-Blend Dataset: Data Structure Overview

## Dataset Description

The **p-Blend Dataset** contains **52-dimensional facial blendshape data** collected from users interacting with various **Virtual Reality (VR) applications**. The dataset aims to study **privacy-preserving techniques** against re-identification attacks in VR environments.

### Facial Blendshape Data

Each data point consists of **52 blendshape values**, representing facial expressions tracked by the **PICO 4 Pro VR headset**. These values capture the movement of facial features like eyes, lips, eyebrows, and jaw.

For a detailed explanation of the blendshape data and its mapping to facial movements, please refer to the official PICO SDK documentation:  
[PICO Unity Face Tracking](https://developer.picoxr.com/document/unity/face-tracking/?v=3.1.0).

## Data File Format

Each file is structured as follows:
1. **Time Step Index**: An integer representing the time step at which the data was recorded.
2. **Blendshape Data**: A list of **52 comma-separated float values** representing the facial blendshape at that time step.
3. **Empty Line**: An empty line separates each time step’s data.

### Example Format:

```
1
0.077173,0.468907,0.019157,0.004002,0.270838,0.050211,0.000000,0.008717,0.031992,0.034745,0.037881,0.058623,0.077173,0.488122,0.002609,0.001506,0.270838,0.129594,0.002143,0.061979,0.096772,0.061783,0.087811,0.048156,0.034397,0.030752,0.086357,0.019308,0.773135,0.179205,0.004002,0.020361,0.001198,0.000072,0.003121,0.020361,0.004002,0.156007,0.511666,0.077173,0.468907,0.019157,0.004002,0.270838,0.050211,0.000000,0.008717,0.031992,0.034745,0.037881,0.058623,0.077173
(empty line)

2
0.111651,0.469502,0.017622,0.004080,0.258907,0.050209,0.000000,0.008715,0.032001,0.035170,0.037524,0.185939,0.111651,0.488632,0.002531,0.001479,0.258907,0.129583,0.002100,0.061978,0.096221,0.061783,0.087977,0.048947,0.034397,0.030765,0.094638,0.019473,0.083612,0.179663,0.004080,0.019686,0.001182,0.000072,0.003030,0.019686,0.004080,0.156711,0.131593,0033769,0.037124,0.104082,0.000597,0.050558,0.185939,0.017622,0.009319,0.009319,0.032674,0.048681,0.000000,0.000000
(empty line)
```

## Directory Structure

The dataset is organized into user, session, and application folders:

```
p_blend_dataset/
└── 1/                          # User 1 data
    ├── session_one/            # First session
    │   ├── Archery/            # Archery app data
    │   │   └── blendshape_data.txt
    │   ├── 360pano/            # 360pano app data
    │   └── ...
    ├── session_two/            # Second session
    │   ├── Archery/            # Archery app data
    │   ├── Sword/              # Sword app data
    │   └── ...
    ├── session_three/          # Third session (long term)
        ├── 360pano/              # Sword app data
        └── Archery/...


### Session Explanation:

- **Session One**: This session includes the user's first interaction with VR applications.
- **Session Two**: The second session represents another round of user interaction. The data collected may reflect more comfort and engagement, as the user becomes familiar with the VR environment. This session includes apps like **Archery**, **Sword**, etc.

- **Session Three**: In some cases, a third session may be included for additional variation in facial expressions as the user continues interacting with different VR applications. The session might contain similar applications as the previous ones but includes more data for analysis.

### Application Folders:

Each application folder contains the **blendshape_data.txt** file, which includes the facial blendshape values for that particular application, recorded at various time steps.

---


## Anonymization and Ethics

- **Anonymization**: The dataset is anonymized by removing any personal identifiers. User folders are named numerically (e.g., `1`, `2`, `3`).
  
- **IRB Approval**: The experiments were approved by the **IRB council of Shandong University**, and all participants provided informed consent.
