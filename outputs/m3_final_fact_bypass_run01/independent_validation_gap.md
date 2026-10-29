# Independent validation gap

Official local validation: 150 scenes / 18 logs; compatible first8 segments: 150; unused compatible validation logs: 0.

The checkpoint embedded training configuration directly identifies nuScenes train and train QA. v4 physical-train scenes are therefore exposed or potentially exposed; changing a round/question order does not create independence. All historical M3/M123 metadata and request records exclude entire logs. Alternate local trainval indexes represent the same physical scenes. A new independently labelled compatible six-camera dataset/log pool and a pre-TEST DEV/protocol freeze are required. No downloads, re-training or old TEST relabelling as new TEST were performed.
