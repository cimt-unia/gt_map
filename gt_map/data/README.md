# Neuroanatomical Atlas Documentation  



**ROI Functional Annotation Guide**

This atlas combines:
- **Cortical regions**: [HCP-MMP1.0](https://doi.org/10.1038/nature18933)
- **Subcortical regions**: [Tian et al. (2020)](https://www.nature.com/articles/s41593-020-00711-6)

Each ROI is annotated with:
- `hemisphere`
- `region_full_name`
- `functional_system` (broad)
- `sub_system` (fine-grained)

Use this guide to interpret connectivity results in your neuroimaging analyses.


<br>

## 1. Visual System

### EarlyRetinotopic (V1, V2, V3)
- **Function**: Low-level visual processing — detects edges, orientation, motion, color.
- **Location**: Occipital lobe.
- **Note**: V1 is the first cortical area to receive visual input (via thalamus).

### Dorsal (V3A, V6, V7, MT/MST, IPS areas)
- **Function**: **“Where/How” pathway** — processes spatial location, motion, depth, and visuomotor coordination (e.g., reaching).
- **Pathway**: Dorsal stream → parietal lobe → guides action.

### Ventral (V4, V8, FFC, PIT, TE, PHC, VVC)
- **Function**: **“What” pathway** — recognizes objects, faces, colors, scenes.
  - **FFC** = Fusiform Face Complex → face recognition  
  - **PHC/PHT** = Parahippocampal areas → scene/place processing
- **Pathway**: Ventral stream → temporal lobe → supports perception and memory.

### Lateral (LO1, LO2, LO3)
- **Function**: Object shape and form processing — bridges dorsal and ventral streams.


<br>

## 2. Motor System

### Primary (Area 4 / M1)
- **Function**: Executes voluntary movements — sends signals to spinal cord to move muscles.
- **Organization**: Somatotopic ("motor homunculus").

### Premotor (6d, 6v, 6a, 55b, 6r)
- **Function**: Plans movements based on external cues (e.g., reaching toward a light).
- **6v** = Ventral premotor → hand/mouth actions, mirror neurons.

### Supplementary (6ma, 6mp, SCEF)
- **Function**: Internally guided actions, sequence planning, bimanual coordination.
- **SCEF** = Supplementary and Cingulate Eye Fields → controls voluntary eye movements.


<br>

## 3. Somatosensory System

### Primary (3a, 3b, 1, 2)
- **Function**: Processes touch, pressure, vibration, proprioception.
  - **3a**: Muscle stretch (proprioception)  
  - **3b/1**: Light touch  
  - **2**: Object shape/texture (stereognosis)

### Association (5m, 5mv, 5L)
- **Function**: Integrates touch + vision + motor → understands object properties during manipulation.

### Operculum (OP1–4)
- **Function**: Pain, temperature, visceral sensation, and speech articulation (OP4 = mouth area).


<br>

## 4. Dorsal Attention Network

### Frontal (FEF)
- **Function**: Top-down control of visual attention — directs eyes to relevant locations.

### Parietal (PEF, IPS1, LIP, VIP, MIP, 7A/7P)
- **Function**: Spatial awareness, eye movement planning, multisensory integration.
- **LIP** = priority map of behaviorally relevant locations.

> **Together**: FEF + Parietal = *"When I decide to look left, these areas activate first."*


<br>

## 5. Ventral Attention Network

### Temporal, Junction, Vestibular (STS, TPOJ, PI, AIP)
- **Function**: Reorients attention to unexpected stimuli (e.g., sudden sound).
  - **STS** = biological motion, voice, lip reading  
  - **AIP** = transforms object shape into hand posture (grasping)  
  - **TPOJ** = tool use, biological motion

> **Note**: Not for sustained attention — for **"Oops, what was that?"** responses.


<br>

## 6. Language System

### Broca (44, 45)
- **Function**: Speech production, grammar, complex syntax.
  - **44** = Pars opercularis → articulation  
  - **45** = Pars triangularis → semantics

### Perisylvian / Parainsular / Retroinsular
- **Function**: Speech perception, phonological processing, auditory-motor integration (part of Wernicke’s network).

### SFL (Superior Frontal Language)
- **Function**: Higher-order language control, narrative comprehension.


<br>

## 7. Default Mode Network (DMN)

> Active during rest, mind-wandering, memory, self-referential thought.

### Prefrontal (9, 10, 8B)
- **Function**: Autobiographical memory, future planning, theory of mind.

### Parietal (PG, PF, IP)
- **Function**: Episodic memory retrieval, semantic integration.

### PosteriorCingulate / Retrosplenial
- **Function**: Memory consolidation, spatial navigation, scene construction.
- **Note**: One of the brain’s most connected and metabolically active hubs.

### Precuneus
- **Function**: Visuospatial imagery, self-consciousness, episodic memory.


<br>

## 8. Cingulo-Opercular Network (CO)

> Stable task control, error monitoring, pain/affect regulation.

### AnteriorCingulate, MidCingulate, SubgenualCingulate
- **dACC**: Conflict monitoring, effortful control  
- **sgACC**: Emotion regulation, mood (implicated in depression)  
- **Midcingulate**: Pain perception, motor pain responses

### Insula, FrontalOperculum
- **Function**: Interoception (internal body state), empathy, disgust, addiction.
- **Anterior Insula** = *"How do I feel right now?"*


<br>

## 9. Limbic System

### Orbitofrontal (47, OFC)
- **Function**: Reward valuation, decision-making, emotional regulation.
- **Key question**: *"Is this worth it?"*

### MedialTemporal (Hippocampus, Entorhinal, Perirhinal)
- **Function**: Episodic memory, spatial navigation.
  - **Hippocampus** = cognitive map  
  - **Entorhinal cortex** = "grid cells"

### Amygdala
- **Function**: Fear processing, emotional salience, threat detection.

### Parahippocampal
- **Function**: Scene/place recognition, contextual memory.


<br>

## 10. Basal Ganglia

> **Modulates** (does not initiate) cortical activity via cortico-basal ganglia-thalamo-cortical loops.

### Striatum (Caudate, Putamen, NAc)
- **Caudate**: Cognitive loops (planning, working memory)  
- **Putamen**: Sensorimotor loops (habit learning, movement scaling)  
- **NAc**: Reward, motivation, addiction

### Pallidum (GPi/GPe)
- **Function**: Output nuclei — inhibit thalamus to **gate movement**.
- **Clinical**: Parkinson’s = dopamine loss → excessive inhibition → akinesia.


<br>

## 11. Thalamus

- **Function**: Relay station for **all sensory input** (except smell) to cortex.
- **Also**: Gating attention — filters irrelevant signals during sleep/focus.


<br>

## 12. Auditory System

### Core (A1)
- **Function**: Basic sound frequency/timing analysis.

### Belt / Association (A4, A5, PBelt, MBelt)
- **Function**: Complex sound processing — speech, music, environmental sounds.
