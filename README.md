## STUDY : Classification of EEG and Sleep Stages Criteria:
* Infrared sleep video and voice database construction project for disease diagnosis using artificial intelligence technology
* Constructed a dataset of infrared sleep video data with over 5000 hours, anonymized and standardized after completion of the necessary tasks.
Seoul National University Hospital participates in research activities to enhance artificial intelligence technology for sleep and speech disorders through the construction of infrared sleep video data and voice data, under the supervision of the institution.

### **Introduction:**

The interpretation of polysomnography (PSG) results follows the guidelines outlined in The AASM Manual for the Scoring of Sleep and Associated Events. Sleep stages, arousals, and associated events are labeled according to the rules defined in the manual. The PSG data, including EEG, EOG, EMG, and respiratory signals, are analyzed in 30-second epochs, and specific criteria are applied for labeling sleep stages and events.

Sleep is a physiological state of altered consciousness where the brain cyclically recovers from accumulated fatigue. It alternates between two main stages: Non-Rapid Eye Movement (NREM) sleep and Rapid Eye Movement (REM) sleep. Typically, REM sleep follows NREM sleep. NREM sleep is further divided into 4 stages (sometimes into 3 stages), constituting 75-80% of the total sleep duration. Stages 1 and 2 represent shallow sleep (Stage 1: an intermediate state between wakefulness and sleep, very light sleep; Stage 2: characterized by slower heart rate, decreased respiration, and dropping body temperature, occupying about half of the total sleep time). Stages 3 and 4 denote deep sleep (a state of complete physical relaxation), followed by REM sleep stage (increased brain activity, dreaming, and rapid eye movements). These stages are categorized based on specific criteria, including brain wave patterns, and each stage has distinct characteristics.
<p align="center">
 <img src="https://github.com/rootofdata/SSU-AI-LAB/assets/86711374/bf61818a-fcbf-49a8-96d3-36ecd793a026",width="250" height="500/">
</p>  

### **Sleep Stages:**
#### **a. Stage W (Wakefulness):**

- Characterized by more than 50% alpha rhythm before sleep onset.
- Occasional eye movements with irregular patterns and increased tension in jaw muscles.

#### **b. Stage N1 (NREM 1 - Shallow Sleep):**
- No alpha waves; lower amplitude and vertex sharp waves in more than 50% of epochs.
- Slow waves (4-7Hz) higher than in Stage W, dominant in posterior head regions, especially in children.

#### **c. Stage N2 (NREM 2 - Shallow Sleep):**

- Presence of sleep spindles and K complexes defines N2 sleep.
- If sleep spindles or K complexes are absent, EEG activity continues.
- Transitions out of N2 occur when switching to Stage W, transitioning to N1 without sleep spindles or K complexes, or transitioning to slow-wave sleep (N3) or REM sleep.

#### **d. Stage N3 (NREM 3 - Deep Sleep):**
- Absence of sleep spindles and eye movements.
- Lower chin muscle tone compared to N2; occasionally slower than REM.

#### **e. Stage R (REM - Dream Sleep):**
- Low amplitude and mixed-frequency EEG, low chin muscle tone, and rapid eye movements characterize REM sleep.
- Scoring Periodic Limb Movements (PLM) and PLM-Associated Arousals:
- LM with a duration of 0.5-10 seconds and starting from resting EMG above 8uV, ending when below 2uV.
- PLM defined as at least 4 LMs with intervals of 5-90 seconds; if multiple LMs occur within 5 seconds, they are considered a single LM.
- PLM-associated arousals occur within 0.5 seconds of LM, with specific EEG criteria and increased chin EMG during REM sleep.

<p align="center">
 <img src="https://github.com/rootofdata/SSU-AI-LAB/assets/86711374/faccce02-9d34-4c39-bfc7-d8a0cdd3cb39",width="250" height="150/">
</p> 

### **Sleep Scoring Principles:**

- If arousals are observed in N2, N3, or R, classify the epoch as N1.
- Increased chin muscle activity during R indicates N1.
- If N3 criteria are not met in the following epoch after N3, reclassify as N2.
- If even partial alpha rhythm is observed during major body movements, classify as W.

<p align="center">
 <img src="https://github.com/rootofdata/SSU-AI-LAB/assets/86711374/6bde9681-c707-4db2-bbcf-bbc8003efd61",width="550" height="300/">
</p> 
<https://m.blog.naver.com/crewblossom/221613672349>

### **Conclusion:**
The PSG interpretation adheres to the principles outlined in The AASM Manual for the Scoring of Sleep and Associated Events. Proper scoring of sleep stages, arousals, and associated events is crucial for accurate diagnosis and treatment planning.


Reference: https://blog.naver.com/cnshs99/222468097699)https://blog.naver.com/cnshs99/222468097699
